from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math

from .module_encoder import TfModel, TextConfig, VisualConfig, AudioConfig
from .until_module import PreTrainedModel, LayerNorm
from .until_module import *
import warnings
import numpy as np
from .losses import *
from perceiver_io.io_processors.postprocessors import AudioPostprocessor, ProjectionPostprocessor, \
    ClassificationPostprocessor
from perceiver_io.io_processors.preprocessors import AudioPreprocessor, ImagePreprocessor, OneHotPreprocessor,FeaturePreprocessor
from perceiver_io.output_queries import FourierQuery, TrainableQuery
from perceiver_io.perceiver import PerceiverIO
from perceiver_io.position_encoding import PosEncodingType
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)



class CARATPreTrainedModel(PreTrainedModel, nn.Module):
    def __init__(self, text_config, visual_config, audio_config,*inputs, **kwargs):
        # utilize bert config as base config
        super(CARATPreTrainedModel, self).__init__(visual_config)
        self.text_config = text_config
        self.visual_config = visual_config
        self.audio_config = audio_config
        self.visual = None
        self.audio = None
        self.text = None

    
    @classmethod
    def from_pretrained(cls, text_model_name, visual_model_name, audio_model_name,
                        state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0
        text_config, _= TextConfig.get_config(text_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        visual_config, _ = VisualConfig.get_config(visual_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        audio_config, _ = AudioConfig.get_config(audio_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        model = cls(text_config, visual_config, audio_config, *inputs, **kwargs)
        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)
        return model


class Normalize(nn.Module):
    def __init__(self, dim):
        super(Normalize, self).__init__()
        self.norm2d = LayerNorm(dim)

    def forward(self, inputs):
        inputs = torch.as_tensor(inputs).float()
        inputs = inputs.view(-1, inputs.shape[-2], inputs.shape[-1])
        output = self.norm2d(inputs)
        return output


def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config
    
class CARAT(CARATPreTrainedModel):
    def __init__(self, text_config, visual_config, audio_config, task_config):
        super(CARAT, self).__init__(text_config, visual_config, audio_config)
        self.task_config = task_config
        self.num_classes = task_config.num_classes
        self.aligned = task_config.aligned
        self.proto_m = task_config.proto_m
        
        # 设置默认值并从task_config中获取perceiver参数
        self.num_latents = getattr(task_config, 'num_latents', 80)
        self.seq_len = getattr(task_config, 'seq_len', 60)
        self.num_self_attends_per_block = getattr(task_config, 'num_self_attends', 6)
        self.num_blocks = getattr(task_config, 'num_blocks', 1)
        self.fourier_num_bands = getattr(task_config, 'fourier_num_bands', 32)
        self.gate_hidden_dim = getattr(task_config, 'gate_hidden_dim', 32)
        self.dropout = getattr(task_config, 'dropout', 0.2)
        
        self.text_norm = Normalize(task_config.text_dim)
        self.visual_norm = Normalize(task_config.video_dim)
        self.audio_norm = Normalize(task_config.audio_dim)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.criterion_cl = SupConLoss()
        self.hard_loss = HardNegativeMiningLoss()
        self.info_cae_loss = info_nce_loss

        self.apply(self.init_weights)

        self.label_attention = MLAttention(self.num_classes, task_config.hidden_size)
      
        self.proj_latent = MLLinear([task_config.hidden_size, task_config.hidden_size//2], task_config.proj_size)
        self.de_proj_text = MLLinear([task_config.proj_size, task_config.hidden_size//2], task_config.hidden_size)
        
        self.va2t = SpectralNoiseGatedAttention(task_config.hidden_size * 3, task_config.hidden_size,self.dropout)

        self.agg = MLLinear([task_config.hidden_size * self.num_classes, task_config.hidden_size], self.num_classes)

        self.text_clf_weight = nn.Parameter(torch.Tensor(self.num_classes, task_config.hidden_size))
        nn.init.kaiming_uniform_(self.text_clf_weight, a=math.sqrt(5))
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('text_pos_protos', torch.zeros(self.num_classes, task_config.proj_size))
        self.register_buffer('text_neg_protos', torch.zeros(self.num_classes, task_config.proj_size))

        self.register_buffer('queue', torch.randn(task_config.moco_queue, task_config.proj_size))
        self.register_buffer("queue_label", torch.randn(task_config.moco_queue, 1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=0)

        if not self.aligned:
            self.a2t_ctc = CTCModule(task_config.audio_dim, 50 if task_config.unaligned_mask_same_length else 500)
            self.v2t_ctc = CTCModule(task_config.video_dim, 50 if task_config.unaligned_mask_same_length else 500)

        # 使用共享的位置编码配置，确保所有模态使用相同的设置
        fourier_pos_kwargs = dict(
            num_bands=self.fourier_num_bands,
            max_resolution=(self.seq_len,),
            sine_only=False,
            concat_pos=True,
        )

        # 修改 Perceiver 配置部分
        self.perceiver = PerceiverIO(
            is_gated=True,
            is_skip_connection=False, 
            num_latents=self.num_latents,  # 使用可配置的num_latents
            num_latent_channels=task_config.hidden_size,
            num_self_attends_per_block=self.num_self_attends_per_block,
            num_blocks=self.num_blocks,
            input_preprocessors={
                'text': FeaturePreprocessor(
                    input_channels=task_config.text_dim,
                    seq_len=self.seq_len,
                    position_encoding_type=PosEncodingType.FOURIER,
                    fourier_position_encoding_kwargs=fourier_pos_kwargs,
                ),
                'visual': FeaturePreprocessor(
                    input_channels=task_config.video_dim,
                    seq_len=self.seq_len,
                    position_encoding_type=PosEncodingType.FOURIER,
                    fourier_position_encoding_kwargs=fourier_pos_kwargs,
                ),
                'audio': FeaturePreprocessor(
                    input_channels=task_config.audio_dim,
                    seq_len=self.seq_len,
                    position_encoding_type=PosEncodingType.FOURIER,
                    fourier_position_encoding_kwargs=fourier_pos_kwargs,
                )
            },
            final_project=True,
            final_project_out_channels=task_config.hidden_size,
            output_queries={
                'unified': TrainableQuery(  
                    output_index_dims=self.num_classes,
                    num_channels=task_config.hidden_size
                )
            }
        )
        
        # 添加gate网络
        self.score_gate = nn.Sequential(
            nn.Linear(2, self.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.gate_hidden_dim, 2),
            nn.Tanh()  # 使用tanh让gate值在[-1,1]范围内
        )
        self.score_gate_three = nn.Sequential(
            nn.Linear(3, self.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.gate_hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # 记录模型配置信息
        logger.info(f"Initialized CARAT with num_latents={self.num_latents}, "
                    f"num_self_attends_per_block={self.num_self_attends_per_block}, "
                    f"num_blocks={self.num_blocks}")

    def dequeue_and_enqueue(self, feats, labels):
        batch_size = feats.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size >= self.task_config.moco_queue:
            self.queue[ptr:,:] = feats[:self.task_config.moco_queue-ptr,:]
            self.queue[:batch_size - self.task_config.moco_queue + ptr,:] = feats[self.task_config.moco_queue-ptr:,:]
            self.queue_label[ptr:, :] = labels[:self.task_config.moco_queue - ptr, :]
            self.queue_label[:batch_size - self.task_config.moco_queue + ptr, :] = labels[self.task_config.moco_queue - ptr:,
                                                                             :]
        else:
            self.queue[ptr:ptr+batch_size, :] = feats
            self.queue_label[ptr:ptr + batch_size, :] = labels
        ptr = (ptr + batch_size) % self.task_config.moco_queue  # move pointer
        self.queue_ptr[0] = ptr


    def get_cl_labels(self, labels, times=1):
        # 只生成 text_cl_labels
        text_labels = torch.zeros_like(labels) + labels
        text_cl_labels = torch.zeros_like(text_labels, dtype=torch.long)
        example_idx, label_idx = torch.where(text_labels >= 0.5)
        text_cl_labels[example_idx, label_idx] = label_idx
        example_idx, label_idx = torch.where(text_labels < 0.5)
        text_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 1

        cl_labels = text_cl_labels.to(torch.int)  # [batch, num_classes]
        if times > 1:
            # repeat along new axis, then flatten
            cl_labels = cl_labels.unsqueeze(1).repeat(1, times, 1)  # [batch, times, num_classes]
            final_cl_labels = cl_labels.view(-1, 1)  # [batch*times*num_classes, 1]
        else:
            final_cl_labels = cl_labels.view(-1, 1)  # [batch*num_classes, 1]
        return final_cl_labels

    def get_cl_mask(self, cl_labels, batch_size):
        mask = torch.eq(cl_labels[:batch_size], cl_labels.T).float()
        neg_mask = torch.ones_like(mask)
        return mask, neg_mask

    def update_protos_dynamic(self, pos_protos, neg_protos, feats, gt_labels):
        b, c = gt_labels.shape[0], gt_labels.shape[1]
        
        # 计算特征与原型的相似度
        feats = feats.view(b, c, -1) 
        pos_sim = F.cosine_similarity(feats, 
                                pos_protos.unsqueeze(0), 
                                dim=-1)  # [b,c]
        neg_sim = F.cosine_similarity(feats, 
                                neg_protos.unsqueeze(0), 
                                dim=-1)  #]
        
        # 动态调整动量系数
        pos_momentum = self.proto_m * (1 + pos_sim) / 2  # 相似度越高，动量越大
        neg_momentum = self.proto_m * (1 + neg_sim) / 2
        
        for i in range(b):
            for j in range(c):
                if gt_labels[i][j] == 1:
                    # 使用动态动量更新正样本原型
                    pos_protos[j] = pos_protos[j] * pos_momentum[i,j] + \
                                   (1 - pos_momentum[i,j]) * feats[i][j]
                else:
                    # 使用动态动量更新负样本原型
                    neg_protos[j] = neg_protos[j] * neg_momentum[i,j] + \
                                   (1 - neg_momentum[i,j]) * feats[i][j]

    def forward(self, text, text_mask, visual, visual_mask, audio, audio_mask,
                label_input, label_mask, groundTruth_labels=None, training=True):
        text = self.text_norm(text)
        visual = self.visual_norm(visual)
        audio = self.audio_norm(audio)
        
        if self.aligned == False:
            visual, v2t_position = self.v2t_ctc(visual)
            audio, a2t_position = self.a2t_ctc(audio)
        
        # 使用 Perceiver 处理特征
        perceiver_inputs = {
            'text': text,
            'visual': visual,
            'audio': audio
        }
        
        perceiver_outputs = self.perceiver(perceiver_inputs,return_intermediates=True)
        cross_attention_outputs = perceiver_outputs['cross_attention']
        # 使用统一的跨模态表示
        latent_outputs = perceiver_outputs['latents']  
        # 分别通过不同的 attention 获取模态特定特征
        unified_feat, text_attention = self.label_attention(cross_attention_outputs, (1 - text_mask).type(torch.bool))
        unified_feat_latent, text_attention = self.label_attention(latent_outputs, (1 - text_mask).type(torch.bool))
        latent_text = self.proj_latent(unified_feat_latent)
        multi_recon_features = perceiver_outputs['outputs']
        text_n = F.normalize(latent_text, p=2, dim=-1)
        text_protos = torch.stack([self.text_pos_protos, self.text_neg_protos])
        text_sim = torch.einsum('bld,nld->bln', text_n, text_protos)
        text_sim = torch.softmax(text_sim, dim=-1)
        if not training:
            text_pos_sim, text_neg_sim = text_sim[:, :, 0], text_sim[:, :, 1]  
            text_pos_mask = (text_pos_sim > text_neg_sim).to(torch.float)
            text_neg_mask = 1 - text_pos_mask
            text_latent_padding = text_pos_mask.unsqueeze(-1) * self.text_pos_protos.unsqueeze(0) + \
                                  text_neg_mask.unsqueeze(-1) * self.text_neg_protos.unsqueeze(0)  
        else:
            text_latent_padding = torch.einsum('bln,nld->bld', text_sim, text_protos)
            
        text_padding = self.de_proj_text(text_latent_padding)
        if training:
            text_aug, l2_loss  = self.va2t(torch.cat([text_padding, unified_feat, unified_feat_latent], dim=-1))
        else:
            text_aug = self.va2t(torch.cat([text_padding, unified_feat, unified_feat_latent], dim=-1))
            l2_loss = 0
        text_clf_out_3 = torch.einsum('bld,ld->bl', text_aug, self.text_clf_weight)
        text_clf_out_1 = torch.einsum('bld,ld->bl', unified_feat, self.text_clf_weight)
        # Add text_clf_out_2 using unified_feat_latent
        text_clf_out_2 = torch.einsum('bld,ld->bl', unified_feat_latent, self.text_clf_weight)
        

        if training:
            latent_aug_text = self.proj_latent(text_aug)
            total_proj = torch.stack([latent_text, latent_aug_text], dim=1)  # [64, 2, 6, 64]
            label_time = 2  # 堆叠了2个，label_time=1
            total_proj = total_proj.view(-1, total_proj.shape[-1])  # [64*2*6, 64] = [768, 64]
            total_proj = F.normalize(total_proj, dim=-1)
            cl_labels = self.get_cl_labels(groundTruth_labels, times=label_time).view(-1).unsqueeze(-1)  # [768, 1]
            text_norm = F.normalize(latent_text.data, dim=-1)
            cl_feats = torch.cat((total_proj, self.queue.clone().detach()), dim=0)
            total_cl_labels = torch.cat((cl_labels, self.queue_label.clone().detach()), dim=0)
            batch_size = cl_feats.shape[0]
            cl_mask, cl_neg_mask = self.get_cl_mask(total_cl_labels, batch_size)
            cl_loss = self.criterion_cl(cl_feats, cl_mask, cl_neg_mask, batch_size)
            self.dequeue_and_enqueue(total_proj, cl_labels)
            self.update_protos_dynamic(self.text_pos_protos, self.text_neg_protos, text_norm, groundTruth_labels)

        clf_out_1 = text_clf_out_1.squeeze(-1)
        clf_out_2 = text_clf_out_2.squeeze(-1)  # Add this line
        clf_out_3 = text_clf_out_3.squeeze(-1)
        predict_scores_clf2 = self.sigmoid(clf_out_2)  # Add this line
        predict_scores_clf3 = self.sigmoid(clf_out_3)
        total_aug = text_aug  # [B, C, D]
        agg_out = self.agg(total_aug.view(total_aug.shape[0], -1))  # [B, C*D] -> [B, num_classes]
        agg_scores = self.sigmoid(agg_out)
        predict_agg_scores = agg_scores  # [B, num_classes]

        # Modify to include clf_out_2 in the adaptive fusion
        predict_final_scores_mean = self.adaptive_weight_fusion_three(
            predict_agg_scores, predict_scores_clf2, predict_scores_clf3
        )
        predict_final_labels_mean = getBinaryTensor(predict_final_scores_mean,
                                                  boundary=self.task_config.binary_threshold)
        predict_scores = predict_final_scores_mean
        predict_labels = predict_final_labels_mean

        if training:
            total_aug_clf_loss = self.bce_loss(agg_out, groundTruth_labels)
            # shuffle_sample_idx shape: [B, C]
            shuffle_sample_idx = torch.zeros(total_aug.shape[0], total_aug.shape[1], dtype=torch.long)
            for m in range(total_aug.shape[1]):
                one_idx = np.random.permutation(total_aug.shape[0])
                shuffle_sample_idx[:, m] = torch.tensor(one_idx, dtype=torch.long)
            # label_idx shape: [1, C] -> [B, C]
            label_idx = torch.arange(self.num_classes).view(1, -1).expand(total_aug.shape[0], total_aug.shape[1])
            # 索引
            shuffle_total_aug = total_aug[shuffle_sample_idx, label_idx]  # [B, C, D]
            shuffle_aug_out = self.agg(shuffle_total_aug.view(total_aug.shape[0], -1))  # [B, num_classes]
            shuffle_gt_labels = groundTruth_labels[shuffle_sample_idx, label_idx]  # [B, C]
            shuffle_aug_clf_loss = self.bce_loss(shuffle_aug_out, shuffle_gt_labels)

        if training:
            all_loss = 0
            clf_loss = self.bce_loss(clf_out_1, groundTruth_labels) * self.task_config.lsr_clf_weight
            # Add clf_out_2 to the loss calculation
            clf_loss += self.bce_loss(clf_out_2, groundTruth_labels) * self.task_config.aug_clf_weight
            clf_loss += self.bce_loss(clf_out_3, groundTruth_labels) * self.task_config.aug_clf_weight
            all_loss += clf_loss

            all_loss += cl_loss * self.task_config.cl_weight

            aug_mse_loss = self.info_cae_loss(text_aug, unified_feat) 
            recon_mse_loss = self.mse_loss(multi_recon_features, unified_feat) 
            all_loss += recon_mse_loss * self.task_config.recon_mse_weight\
                        + aug_mse_loss * self.task_config.aug_mse_weight

            all_loss += total_aug_clf_loss * self.task_config.total_aug_clf_weight
            all_loss += shuffle_aug_clf_loss * self.task_config.shuffle_aug_clf_weight 
            all_loss += l2_loss * self.task_config.l2_weight
            # all_loss += hard_loss * 0.1
            return all_loss, predict_labels, groundTruth_labels, predict_scores
        else:

            return predict_labels, groundTruth_labels, predict_scores

    def adaptive_weight_fusion(self, scores1, scores2):
        """带学习gate的自适应权重融合
        Args:
            scores1: 第一组预测分数 [batch_size, num_classes]
            scores2: 第二组预测分数 [batch_size, num_classes] 
        Returns:
            融合后的分数 [batch_size, num_classes]
        """
        # 计算置信度
        confidence1 = torch.abs(scores1 - 0.5)
        confidence2 = torch.abs(scores2 - 0.5)
        
        # 拼接两个分数作为gate网络的输入
        gate_input = torch.stack([scores1, scores2], dim=-1)  # [batch_size, 2]
        
        # 获取学习的gate系数
        gate_weights = self.score_gate(gate_input)  # [batch_size, 2]
        
        # 结合置信度和gate权重
        weights = torch.stack([confidence1, confidence2], dim=-1)  # [batch_size, 2]
        weights = weights * (1 + gate_weights)  # gate调制置信度权重
        
        # softmax归一化
        weights = F.softmax(weights, dim=-1)
        
        # 加权融合
        final_scores = weights[...,0] * scores1 + weights[...,1] * scores2
        
        return final_scores

    def adaptive_weight_fusion_three(self, scores1, scores2, scores3):
        """三组分数的自适应权重融合
        Args:
            scores1: 第一组预测分数 [batch_size, num_classes]
            scores2: 第二组预测分数 [batch_size, num_classes]
            scores3: 第三组预测分数 [batch_size, num_classes]
        Returns:
            融合后的分数 [batch_size, num_classes]
        """
        # 计算每组分数的置信度
        confidence1 = torch.abs(scores1 - 0.5)
        confidence2 = torch.abs(scores2 - 0.5)
        confidence3 = torch.abs(scores3 - 0.5)

        # 拼接三个分数作为gate网络的输入
        gate_input = torch.stack([scores1, scores2, scores3], dim=-1)  # [batch_size, num_classes, 3]
        
        # 使用预定义的网络
        gate_weights = self.score_gate_three(gate_input)
        
        # 结合置信度和gate权重
        weights = torch.stack([confidence1, confidence2, confidence3], dim=-1)  # [batch_size, num_classes, 3]
        weights = weights * gate_weights  # gate调制置信度权重

        # softmax归一化
        weights = F.softmax(weights, dim=-1)

        # 加权融合
        final_scores = weights[...,0] * scores1 + weights[...,1] * scores2 + weights[...,2] * scores3

        return final_scores


