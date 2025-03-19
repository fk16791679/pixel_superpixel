import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PixelContrastiveLearning(nn.Module):
    def __init__(self, feature_dim=2048, temperature=0.1, base_temperature=0.07, 
                 max_samples=256, superpixel_mask=False, similarity_threshold=0.7):  # 进一步减小max_samples
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
        self.loss_weight = 0.1  # 添加损失权重
        
        # 投影头 - 用于将特征映射到对比学习空间
        self.projection = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1)
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
        target_labels = F.interpolate(target_labels.unsqueeze(1).float(), 
                                    size=(feat_h, feat_w), 
                                    mode='nearest').squeeze(1).long()
        # 转换特征形状为 (B, H*W, C)
        feats_flatten = proj_feats.permute(0, 2, 3, 1)  # [B, H, W, 256]
        feats_flatten = feats_flatten.reshape(batch_size, feat_h * feat_w, 256)
        pixel_weights = pixel_weights.permute(0, 2, 3, 1)  # [B, H, W, 1]
        pixel_weights = pixel_weights.reshape(batch_size, feat_h * feat_w, 1)
        
        # 归一化特征
        feats_flatten = F.normalize(feats_flatten, p=2, dim=-1)
        
        # 转换标签形状为 (B, H*W)
        labels_flatten = target_labels.reshape(batch_size, -1)
        superpixel_flatten = None
        # 转换超像素索引形状为 (B, H*W)
        if self.superpixel_mask and superpixel_indice is not None:
            superpixel_flatten = superpixel_indice.reshape(batch_size, -1)
        
        # 计算像素对比损失
        total_loss = 0.0
        n_loss = 0
        
        for b in range(batch_size):
            # 获取有效区域（TLS区域）
            valid_mask = (labels_flatten[b] == 1)
            if valid_mask.sum() == 0:
                continue
            
            # 获取TLS区域的特征并限制最大数量
            valid_indices = valid_mask.nonzero().squeeze()
            if valid_indices.numel() > self.max_samples:
                idx = torch.randperm(valid_indices.numel(), device=valid_indices.device)[:self.max_samples]
                valid_indices = valid_indices[idx]
                
            tls_feats = feats_flatten[b, valid_indices]
            tls_weights = pixel_weights[b, valid_indices]
            
            # 分批计算相似度矩阵以节省内存
            chunk_size = min(64, tls_feats.shape[0])  # 确保chunk_size不超过特征数量
            n_chunks = (tls_feats.shape[0] + chunk_size - 1) // chunk_size
            sim_matrix = torch.zeros(tls_feats.shape[0], tls_feats.shape[0], 
                                   device=tls_feats.device)
            
            # 检查特征维度
            if tls_feats.shape[0] == 0:
                continue
            
            # 确保所有张量在同一设备上
            device = feats.device
            if labels is not None:
                labels = labels.to(device)
            if pseudo_labels is not None:
                pseudo_labels = pseudo_labels.to(device)
                
            # 使用torch.cuda.amp进行混合精度训练
            with torch.cuda.amp.autocast(enabled=True):
                for i in range(n_chunks):
                    start_i = i * chunk_size
                    end_i = min(start_i + chunk_size, tls_feats.shape[0])
                    chunk_i = tls_feats[start_i:end_i]
                    
                    # 计算当前chunk的相似度
                    chunk_sim = torch.matmul(chunk_i, tls_feats.t())
                    sim_matrix[start_i:end_i] = chunk_sim
                    
                    # 定期清理缓存
                    if i % 2 == 0:
                        torch.cuda.empty_cache()
            
            # 定义正负样本
            pos_mask = (sim_matrix > self.similarity_threshold)
            if self.superpixel_mask and superpixel_indice is not None:
                # 获取当前batch的超像素索引
                curr_superpixel = superpixel_flatten[b, valid_indices]
                # 创建超像素约束掩码
                superpixel_constraint = (curr_superpixel.unsqueeze(1) == curr_superpixel.unsqueeze(0))
                # 结合特征相似度和超像素约束
                pos_mask = pos_mask & superpixel_constraint
            
            identity_mask = torch.eye(tls_feats.shape[0], dtype=torch.bool, device=tls_feats.device)
            pos_mask.masked_fill_(identity_mask, False)
            
            if pos_mask.sum() == 0:
                continue
                
            # 分批计算损失
            sim_matrix = sim_matrix / self.temperature
            exp_sim = torch.exp(sim_matrix)
            del sim_matrix  # 释放内存
            
            denominator = exp_sim.sum(dim=1)
            pos_pairs = pos_mask.nonzero()
            
            loss = 0
            total_weights_sum = tls_weights.sum()  # Calculate sum of weights once
            n_pairs = pos_pairs.shape[0]
            pair_chunks = (n_pairs + chunk_size - 1) // chunk_size
            
            for i in range(pair_chunks):
                start_i = i * chunk_size
                end_i = min(start_i + chunk_size, n_pairs)
                chunk_pairs = pos_pairs[start_i:end_i]
                
                row_idx = chunk_pairs[:, 0]
                col_idx = chunk_pairs[:, 1]
                
                pos_sim = exp_sim[row_idx, col_idx]
                pos_weights_chunk = tls_weights[row_idx] * tls_weights[col_idx]
                
                log_prob = torch.log(pos_sim / denominator[row_idx])
                loss += -(self.temperature / self.base_temperature) * (pos_weights_chunk * log_prob).sum()
            
            loss = loss / total_weights_sum  # Use the total weights sum
            total_loss += loss
            n_loss += 1
            
            del exp_sim  # 释放内存

        if n_loss > 0:
            return (total_loss / n_loss) * self.loss_weight  # 应用损失权重
        else:
            return torch.tensor(0.0, device=feats.device, requires_grad=True)
