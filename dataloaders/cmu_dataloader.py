# %%
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import math

emotion_dict = {4:0, 5:1, 6:2, 7:3, 8:4, 9:5}
class AlignedMoseiDataset(Dataset):
    def __init__(self, data_path, data_type, args):
        self.data_path = data_path
        self.data_type = data_type
        self.args = args

        # 加载基础数据以确定数据集大小
        self.visual, self.audio, \
            self.text, self.labels = self._get_data(self.data_type)
        self.dataset_len = len(self.labels) # 获取数据集长度

        # 初始化存储缺失决策的列表
        self.missing_decisions = ['none'] * self.dataset_len

        # 仅在测试集上预计算缺失决策
        if self.data_type in ['test',  'valid']:
            missing_prob = getattr(args, 'missing_prob', 0.0)
            seed = getattr(args, 'seed', 42)
            if missing_prob > 0:
                print(f"INFO: Pre-computing missing modality decisions for {self.dataset_len} AlignedMoseiDataset  {self.data_type} samples with prob: {missing_prob}, seed: {seed}")
                # 使用固定种子初始化 RNG 用于预计算
                rng = np.random.RandomState(seed)
                modalities = ['text', 'visual', 'audio']
                for i in range(self.dataset_len):
                    # 决定这个样本是否缺失模态
                    if rng.rand() < missing_prob:
                        # 如果缺失，决定缺失哪个模态
                        self.missing_decisions[i] = rng.choice(modalities)
                print("INFO: Pre-computing finished.")
            # else: missing_decisions 保持为 'none'

    def _get_data(self, data_type):
        data = torch.load(self.data_path,weights_only=False)
        data = data[data_type]
        visual = data['src-visual']
        audio = data['src-audio']
        text = data['src-text']
        labels = data['tgt']
        return visual, audio, text, labels

    def _get_text(self, index):
        text = self.text[index]
        text_mask = np.ones(text.shape[0], dtype=np.int64)
        return text, text_mask

    def _get_visual(self, index):
        visual = self.visual[index]
        visual_mask = np.ones(visual.shape[0], dtype=np.int64)
        return visual, visual_mask

    def _get_audio(self, index):
        audio = self.audio[index]
        audio[audio == -np.inf] = 0
        audio_mask = np.ones(audio.shape[0], dtype=np.int64)
        return audio, audio_mask

    def _get_labels(self, index):
        label_list = self.labels[index]
        label = np.zeros(6, dtype=np.float32)
        filter_label = label_list[1:-1]
        for emo in filter_label:
            if emo in emotion_dict:
                label[emotion_dict[emo]] = 1
        return label

    def _get_label_input(self):
        labels_embedding = np.arange(6)
        labels_mask = np.ones(labels_embedding.shape[0], dtype=np.int64)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)
        return labels_embedding, labels_mask

    def __len__(self):
        # 使用预存的长度
        return self.dataset_len

    def __getitem__(self, index):
        # 获取原始数据
        text, text_mask = self._get_text(index)
        visual, visual_mask = self._get_visual(index)
        audio, audio_mask = self._get_audio(index)
        label = self._get_labels(index)

        # 应用预计算的缺失决策 (仅对测试集有效，因为非测试集的 missing_decisions 全是 'none')
        modality_to_drop = self.missing_decisions[index]

        if modality_to_drop == 'text':
            text = np.zeros_like(text)
            text_mask = np.zeros_like(text_mask)
        elif modality_to_drop == 'visual':
            visual = np.zeros_like(visual)
            visual_mask = np.zeros_like(visual_mask)
        elif modality_to_drop == 'audio':
            audio = np.zeros_like(audio)
            audio_mask = np.zeros_like(audio_mask)
        # 如果 modality_to_drop 是 'none', 则不执行任何操作

        return text, text_mask, visual, visual_mask, \
            audio, audio_mask, label


class UnAlignedMoseiDataset(Dataset):
    def __init__(self, data_path, data_type, args):
        self.data_path = data_path
        self.data_type = data_type
        self.args = args

        # 加载基础数据以确定数据集大小
        self.visual, self.audio, \
            self.text, self.labels = self._get_data(self.data_type)
        self.dataset_len = len(self.labels) # 获取数据集长度

        # 初始化存储缺失决策的列表
        self.missing_decisions = ['none'] * self.dataset_len

        # 仅在测试集上预计算缺失决策
        if self.data_type == 'test':
            missing_prob = getattr(args, 'missing_prob', 0.0)
            seed = getattr(args, 'seed', 42)
            if missing_prob > 0:
                print(f"INFO: Pre-computing missing modality decisions for {self.dataset_len} UnAlignedMoseiDataset (test) samples with prob: {missing_prob}, seed: {seed}")
                # 使用固定种子初始化 RNG 用于预计算
                rng = np.random.RandomState(seed)
                modalities = ['text', 'visual', 'audio']
                for i in range(self.dataset_len):
                    # 决定这个样本是否缺失模态
                    if rng.rand() < missing_prob:
                        # 如果缺失，决定缺失哪个模态
                        self.missing_decisions[i] = rng.choice(modalities)
                print("INFO: Pre-computing finished.")
            # else: missing_decisions 保持为 'none'

    def _get_data(self, data_type):
        label_data = torch.load(self.data_path)
        label_data = label_data[data_type]
        try:
            with open('./data/mosei_senti_data_noalign.pkl', 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Could not find './data/mosei_senti_data_noalign.pkl'. Please check the path.")

        data = data[data_type]
        visual = data['vision']
        audio = data['audio']
        text = data['text']
        labels = label_data['tgt']
        return visual, audio, text, labels

    def _get_text(self, index):
        text = self.text[index]
        if not isinstance(text, np.ndarray): text = np.array(text)
        text_mask = np.ones(text.shape[0], dtype=np.int64)
        return text, text_mask

    def _get_visual(self, index):
        visual = self.visual[index]
        if not isinstance(visual, np.ndarray): visual = np.array(visual)

        if getattr(self.args, 'unaligned_mask_same_length', False):
            visual_mask = np.ones(50, dtype=np.int64)
        else:
            visual_mask = np.ones(visual.shape[0], dtype=np.int64)
        return visual, visual_mask

    def _get_audio(self, index):
        audio = self.audio[index]
        if not isinstance(audio, np.ndarray): audio = np.array(audio)

        audio[audio == -np.inf] = 0
        if getattr(self.args, 'unaligned_mask_same_length', False):
            audio_mask = np.ones(50, dtype=np.int64)
        else:
            audio_mask = np.ones(audio.shape[0], dtype=np.int64)
        return audio, audio_mask

    def _get_labels(self, index):
        label_list = self.labels[index]
        label = np.zeros(6, dtype=np.float32)
        filter_label = label_list[1:-1]
        for emo in filter_label:
            if emo in emotion_dict:
                label[emotion_dict[emo]] = 1
        return label

    def _get_label_input(self):
        labels_embedding = np.arange(6)
        labels_mask = np.ones(labels_embedding.shape[0], dtype=np.int64)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)
        return labels_embedding, labels_mask

    def __len__(self):
        # 使用预存的长度
        return self.dataset_len

    def __getitem__(self, index):
        # 获取原始数据
        text, text_mask = self._get_text(index)
        visual, visual_mask = self._get_visual(index)
        audio, audio_mask = self._get_audio(index)
        label = self._get_labels(index)

        # 应用预计算的缺失决策 (仅对测试集有效)
        modality_to_drop = self.missing_decisions[index]

        if modality_to_drop == 'text':
            text = np.zeros_like(text)
            text_mask = np.zeros_like(text_mask)
        elif modality_to_drop == 'visual':
            visual = np.zeros_like(visual)
            # 根据 unaligned_mask_same_length 处理视觉掩码
            if getattr(self.args, 'unaligned_mask_same_length', False):
                visual_mask = np.zeros(50, dtype=np.int64)
            else:
                visual_mask = np.zeros_like(visual_mask)
        elif modality_to_drop == 'audio':
            audio = np.zeros_like(audio)
            # 根据 unaligned_mask_same_length 处理音频掩码
            if getattr(self.args, 'unaligned_mask_same_length', False):
                audio_mask = np.zeros(50, dtype=np.int64)
            else:
                audio_mask = np.zeros_like(audio_mask)
        # 如果 modality_to_drop 是 'none', 则不执行任何操作

        return text, text_mask, visual, visual_mask, \
            audio, audio_mask, label