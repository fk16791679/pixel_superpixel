import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PixelContrastiveLearning(nn.Module):
    def __init__(self, feature_dim=256, temperature=0.1, base_temperature=0.07, 
                 max_samples=1024, superpixel_mask=True, similarity_threshold=0.7):
        """
        像素级对比学习模块
        
        Args:
            feature_dim: 特征维度
            temperature: 对比学习温度参数
            base_temperature: 基础温度参数
            max_samples: 每类最大采样像素数
            superpixel_mask: 是否使用超像素作为空间约束
            similarity_threshold: 特征相似度阈值，用于动态选择正样本
        """
        super(PixelContrastiveLearning, self).__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.max_samples = max_samples
        self.superpixel_mask = superpixel_mask
        self.similarity_threshold = similarity_threshold
        
        # 投影头 - 用于将特征映射到对比学习空间
        self.projection = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        )
        
        # 特征注意力模块 - 学习每个像素的权重
        self.pixel_attention = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, feats, labels=None, superpixel_indice=None, pseudo_labels=None):
        """
        执行像素级对比学习
        
        Args:
            feats: 特征图 (B, C, H, W)
            labels: 真实标签或粗标注 (B, H, W)
            superpixel_indice: 超像素索引图 (B, H, W)
            pseudo_labels: 伪标签 (B, H, W)，可选，用于迭代修正
        
        Returns:
            对比学习损失
        """
        batch_size, feat_dim, feat_h, feat_w = feats.size()
        
        # 应用投影头
        proj_feats = self.projection(feats)
        
        # 生成像素注意力权重
        pixel_weights = self.pixel_attention(feats)
        
        # 使用真实标签或伪标签
        if pseudo_labels is not None and torch.rand(1).item() < 0.5:  # 50%概率使用伪标签
            target_labels = pseudo_labels
        else:
            target_labels = labels
        
        # 转换特征形状为 (B, H*W, C)
        feats_flatten = proj_feats.permute(0, 2, 3, 1).reshape(batch_size, -1, feat_dim)
        pixel_weights = pixel_weights.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)
        
        # 归一化特征
        feats_flatten = F.normalize(feats_flatten, p=2, dim=-1)
        
        # 转换标签形状为 (B, H*W)
        labels_flatten = target_labels.reshape(batch_size, -1)
        
        # 转换超像素索引形状为 (B, H*W)
        if self.superpixel_mask and superpixel_indice is not None:
            superpixel_flatten = superpixel_indice.reshape(batch_size, -1)
        
        # 计算像素对比损失
        total_loss = 0.0
        n_loss = 0
        
        for b in range(batch_size):
            # 获取有效区域（TLS区域）
            valid_mask = (labels_flatten[b] == 1)  # 假设TLS区域标记为1
            if valid_mask.sum() == 0:
                continue
            
            # 获取TLS区域的特征
            tls_feats = feats_flatten[b, valid_mask]
            tls_weights = pixel_weights[b, valid_mask]
            
            # 随机采样以限制计算复杂度
            if tls_feats.shape[0] > self.max_samples:
                idx = np.random.choice(tls_feats.shape[0], self.max_samples, replace=False)
                tls_feats = tls_feats[idx]
                tls_weights = tls_weights[idx]
                if self.superpixel_mask:
                    tls_superpixel = superpixel_flatten[b, valid_mask][idx]
            elif self.superpixel_mask:
                tls_superpixel = superpixel_flatten[b, valid_mask]
            
            # 计算特征相似度矩阵
            sim_matrix = torch.matmul(tls_feats, tls_feats.t())
            
            # 定义正负样本
            if self.superpixel_mask:
                # 空间约束：同一超像素区域内为潜在正样本
                superpixel_mask = (tls_superpixel.unsqueeze(0) == tls_superpixel.unsqueeze(1))
                
                # 特征约束：高相似度特征为正样本
                sim_mask = (sim_matrix > self.similarity_threshold)
                
                # 同时满足空间和特征约束的为正样本
                pos_mask = superpixel_mask & sim_mask
            else:
                # 仅使用特征相似度约束
                pos_mask = (sim_matrix > self.similarity_threshold)
            
            # 移除自身的对角线
            identity_mask = torch.eye(tls_feats.shape[0], dtype=torch.bool, device=tls_feats.device)
            pos_mask.masked_fill_(identity_mask, False)
            
            # 检查是否有足够的正样本对
            if pos_mask.sum() == 0:
                continue
            
            # 计算正样本对的损失
            pos_sim = sim_matrix[pos_mask]
            pos_weights = (tls_weights * tls_weights.t())[pos_mask]
            
            # 应用温度参数
            pos_sim /= self.temperature
            
            # log(exp(pos_sim) / sum(exp(sim)))
            nominator = torch.exp(pos_sim)
            denominator = torch.sum(torch.exp(sim_matrix / self.temperature), dim=1)
            
            # 计算加权对比损失
            contrastive_log_prob = torch.log(nominator / denominator.unsqueeze(1)[pos_mask])
            loss = -(self.temperature / self.base_temperature) * (pos_weights * contrastive_log_prob).sum() / pos_weights.sum()
            
            total_loss += loss
            n_loss += 1
        
        if n_loss > 0:
            return total_loss / n_loss
        else:
            # 如果没有有效样本，返回零张量
            return torch.tensor(0.0, device=feats.device, requires_grad=True)
