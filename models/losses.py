import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeMiningLoss(nn.Module):
    """难例挖掘损失
    针对展平后的特征进行难例挖掘
    """
    def __init__(self, temperature=0.07, k_hard=3):
        super().__init__()
        self.temperature = temperature
        self.k_hard = k_hard
        
    def forward(self, features, labels, batch_size=-1):
        """
        Args:
            features: 展平的特征向量 [N, feature_dim]，包含当前批次和历史特征
            labels: 多标签 [batch_size, num_classes]
            batch_size: 当前批次大小
        Returns:
            loss: 难例挖掘损失
        """
        if batch_size == -1:
            batch_size = labels.shape[0]
            
        # 获取当前批次的特征作为anchors
        anchors = features[:batch_size]  # [batch_size, feature_dim]
        
        # 计算anchor与所有样本的相似度矩阵
        sim_matrix = torch.matmul(anchors, features.T)  # [batch_size, N]
        sim_matrix = torch.div(sim_matrix, self.temperature)
        
        # 构建负样本mask
        # 扩展labels以匹配memory bank
        expanded_labels = labels.repeat_interleave(6, dim=0)  # 因为每个样本重复了6次
        neg_mask = (1 - torch.mm(labels, expanded_labels.t()))  # [batch_size, N]
        
        # 移除自身的相似度
        self_mask = torch.zeros_like(neg_mask)
        self_mask[:, :batch_size] = torch.eye(batch_size)
        neg_mask = neg_mask * (1 - self_mask)
        
        # 在负样本中找出最相似的k个作为难例
        neg_sim = sim_matrix * neg_mask - (1 - neg_mask) * 1e12
        hard_idx = torch.topk(neg_sim, k=min(self.k_hard, neg_sim.shape[1]-1), dim=1)[1]  # [batch_size, k]
        
        # 构建anchor-hard pairs
        anchors = F.normalize(anchors, dim=1)
        hard_negatives = F.normalize(features[hard_idx], dim=2)  # [batch_size, k, feature_dim]
        
        # 计算anchor和难例之间的相似度
        hard_sim = torch.einsum('bd,bkd->bk', anchors, hard_negatives)  # [batch_size, k]
        hard_sim = hard_sim / self.temperature
        
        # 希望降低与难例的相似度
        loss = torch.mean(F.softplus(hard_sim))
        
        return loss


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask, neg_mask=None, batch_size=-1, device=None, other_features=None):

        if mask is not None:
            mask = mask.float().detach()
            if other_features is None:
                anchor_dot_contrast = torch.div(
                    torch.matmul(features[:batch_size], features.T),
                    self.temperature)
            else:
                anchor_dot_contrast = torch.div(
                    torch.matmul(features[:batch_size], other_features.T),
                    self.temperature)
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            if neg_mask is None:
                logits_mask = torch.ones_like(mask)
            else:
                logits_mask = torch.scatter(
                    neg_mask,
                    1,
                    torch.arange(batch_size).view(-1, 1).to(neg_mask.device),
                    0
                )
            mask = mask * logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size * 2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss
    
def info_nce_loss(features1, features2, temperature=0.07):
    """
    计算 InfoNCE loss，先将特征展平成 [B, T*D] 再计算
    Args:
        features1: 第一组特征 [B, T, D]
        features2: 第二组特征 [B, T, D]
        temperature: 温度参数
    Returns:
        loss: InfoNCE loss
    """
    B, T, D = features1.shape
    feat1 = F.normalize(features1.reshape(B, T * D), dim=1)  # [B, T*D]
    feat2 = F.normalize(features2.reshape(B, T * D), dim=1)  # [B, T*D]

    similarity = torch.matmul(feat1, feat2.T) / temperature  # [B, B]
    labels = torch.arange(B, device=features1.device)
    loss = F.cross_entropy(similarity, labels)
    return loss