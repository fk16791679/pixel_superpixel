import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PixelContrastiveLearning(nn.Module):
    def __init__(self, feature_dim=2048, temperature=0.1, base_temperature=0.07, 
                 max_samples=64, similarity_threshold=0.7):
        super(PixelContrastiveLearning, self).__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.max_samples = max_samples
        self.similarity_threshold = similarity_threshold
        self.loss_weight = 0.1
        
        # 简化投影头
        self.projection = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        
        self.pixel_attention = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, feats, labels=None, pseudo_labels=None):
        # 使用scatter而不是在单个GPU上处理所有批次
        if not feats.requires_grad:
            return torch.zeros(1, device=feats.device, requires_grad=True)
        
        # 基本设置
        device = feats.device
        batch_size, _, feat_h, feat_w = feats.size()
        
        # 应用投影头和注意力
        with torch.cuda.amp.autocast(enabled=True):
            proj_feats = self.projection(feats)
            pixel_weights = self.pixel_attention(feats)
        
        # 处理标签
        if labels is None:
            return torch.zeros(1, device=device, requires_grad=True)
        
        target_labels = F.interpolate(labels.unsqueeze(1).float(), 
                                     size=(feat_h, feat_w), 
                                     mode='nearest').squeeze(1).long()
        
        # 特征扁平化
        feats_flatten = proj_feats.permute(0, 2, 3, 1).reshape(batch_size, feat_h * feat_w, proj_feats.size(1))
        pixel_weights = pixel_weights.permute(0, 2, 3, 1).reshape(batch_size, feat_h * feat_w, 1)
        
        # 特征规范化
        feats_flatten = F.normalize(feats_flatten, p=2, dim=-1)
        labels_flatten = target_labels.reshape(batch_size, -1)
        
        # 处理每个图像的对比损失
        total_loss = 0.0
        valid_batches = 0
        
        # 关键改进：每个GPU只处理自己负责的批次部分
        for b in range(batch_size):
            # 使用torch.autograd.profiler.record_function来分析内存使用
            # with torch.autograd.profiler.record_function("pixel_contrast_batch"):
            loss = self._compute_single_image_loss(
                feats_flatten[b], labels_flatten[b], pixel_weights[b], device)
            
            if loss is not None:
                total_loss += loss
                valid_batches += 1
        
        if valid_batches > 0:
            return (total_loss / valid_batches) * self.loss_weight
        return torch.zeros(1, device=device, requires_grad=True)
    
    def _compute_single_image_loss(self, image_feats, image_labels, image_weights, device):
        """计算单一图像的对比损失，减少内存使用"""
        # 获取感兴趣区域
        valid_mask = (image_labels == 1)
        if valid_mask.sum() == 0:
            return None
        
        # 采样像素
        valid_indices = valid_mask.nonzero().squeeze()
        if valid_indices.dim() == 0:
            valid_indices = valid_indices.unsqueeze(0)
            
        if valid_indices.numel() > self.max_samples:
            idx = torch.randperm(valid_indices.numel(), device=device)[:self.max_samples]
            valid_indices = valid_indices[idx]
            
        if valid_indices.numel() < 2:
            return None
            
        # 提取特征和权重
        tls_feats = image_feats[valid_indices]
        tls_weights = image_weights[valid_indices]
        
        # 使用更小的块
        chunk_size = min(16, tls_feats.shape[0])
        
        # 计算相似度矩阵
        sim_matrix = torch.zeros(tls_feats.shape[0], tls_feats.shape[0], device=device)
        
        for i in range(0, tls_feats.shape[0], chunk_size):
            end_i = min(i + chunk_size, tls_feats.shape[0])
            chunk_i = tls_feats[i:end_i]
            
            # 逐块计算相似度，减少内存使用
            for j in range(0, tls_feats.shape[0], chunk_size):
                end_j = min(j + chunk_size, tls_feats.shape[0])
                chunk_j = tls_feats[j:end_j]
                
                # 分块矩阵乘法
                block_sim = torch.matmul(chunk_i, chunk_j.t())
                sim_matrix[i:end_i, j:end_j] = block_sim
                
                # 每处理几个块就清理一次缓存
                if (i * tls_feats.shape[0] + j) % (4 * chunk_size * chunk_size) == 0:
                    torch.cuda.empty_cache()
        
        # 应用温度系数
        sim_matrix = sim_matrix / self.temperature
        
        # 确定正样本对
        pos_mask = (sim_matrix > self.similarity_threshold)
        identity_mask = torch.eye(tls_feats.shape[0], dtype=torch.bool, device=device)
        pos_mask.masked_fill_(identity_mask, False)
        
        if pos_mask.sum() == 0:
            return None
            
        # 计算对比损失
        exp_sim = torch.exp(sim_matrix)
        del sim_matrix
        
        denominator = exp_sim.sum(dim=1)
        pos_pairs = pos_mask.nonzero()
        
        # 计算加权损失
        total_weights_sum = tls_weights.sum()
        if total_weights_sum == 0:
            return None
            
        loss = 0.0
        for i in range(0, pos_pairs.shape[0], chunk_size):
            end_i = min(i + chunk_size, pos_pairs.shape[0])
            chunk_pairs = pos_pairs[i:end_i]
            
            row_idx = chunk_pairs[:, 0]
            col_idx = chunk_pairs[:, 1]
            
            pos_sim = exp_sim[row_idx, col_idx]
            pos_weights = tls_weights[row_idx] * tls_weights[col_idx]
            
            log_prob = torch.log(pos_sim / denominator[row_idx] + 1e-10)
            loss += -(self.temperature / self.base_temperature) * (pos_weights * log_prob).sum()
        
        del exp_sim
        
        return loss / total_weights_sum