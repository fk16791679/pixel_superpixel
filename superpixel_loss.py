import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

class SuperpixelContrastiveLearning(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=128, output_dim=64, 
                 temperature=0.1, threshold=0.7, dropout=0.1):
        """
        超像素级对比学习模块
        
        Args:
            feature_dim: 输入特征维度
            hidden_dim: 隐层特征维度
            output_dim: 输出特征维度
            temperature: 对比学习温度参数
            threshold: 高置信度样本选择阈值
            dropout: Dropout率
        """
        super(SuperpixelContrastiveLearning, self).__init__()
        
        # 特征转换网络 - 简化的图卷积结构
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 双分支投影头
        self.projector1 = nn.Linear(hidden_dim, output_dim)
        self.projector2 = nn.Linear(hidden_dim, output_dim)
        
        self.temperature = temperature
        self.threshold = threshold
        
        # 用于TLS分割的类别数(TLS区域和背景)
        self.n_classes = 2
    
    def aggregate_superpixel_features(self, features, superpixel_indices):
        """
        聚合每个超像素区域内的特征
        
        Args:
            features: 像素级特征 [B, C, H, W]
            superpixel_indices: 超像素索引图 [B, H, W]
            
        Returns:
            超像素级特征 [B, N, C] 其中N是每个图像中的超像素数量
        """
        batch_size = features.shape[0]
        feature_dim = features.shape[1]
        all_sp_features = []
        all_sp_counts = []
        
        for b in range(batch_size):
            # 获取当前批次的特征和超像素索引
            feat = features[b]  # [C, H, W]
            sp_idx = superpixel_indices[b]  # [H, W]
            
            # 计算超像素数量
            n_sp = sp_idx.max().item() + 1
            
            # 初始化超像素特征和计数
            sp_features = torch.zeros(n_sp, feature_dim, device=features.device)
            sp_counts = torch.zeros(n_sp, 1, device=features.device)
            
            # 将特征重塑为[C, H*W]
            feat_flat = feat.reshape(feature_dim, -1)
            sp_idx_flat = sp_idx.reshape(-1)
            
            # 聚合特征
            for i in range(n_sp):
                mask = (sp_idx_flat == i)
                if mask.sum() > 0:
                    sp_features[i] = feat_flat[:, mask].mean(dim=1)
                    sp_counts[i] = mask.sum()
            
            all_sp_features.append(sp_features)
            all_sp_counts.append(sp_counts)
            
        return all_sp_features, all_sp_counts
    
    def build_adjacency_matrix(self, superpixel_indices):
        """
        构建超像素邻接矩阵
        
        Args:
            superpixel_indices: 超像素索引图 [B, H, W]
            
        Returns:
            邻接矩阵列表 [B, N, N]
        """
        batch_size = superpixel_indices.shape[0]
        all_adj = []
        
        for b in range(batch_size):
            sp_idx = superpixel_indices[b]  # [H, W]
            h, w = sp_idx.shape
            n_sp = sp_idx.max().item() + 1
            
            # 初始化邻接矩阵
            adj = torch.zeros(n_sp, n_sp, device=sp_idx.device)
            
            # 计算水平方向的邻接关系
            for i in range(h):
                for j in range(w-1):
                    sp1 = sp_idx[i, j].item()
                    sp2 = sp_idx[i, j+1].item()
                    if sp1 != sp2:
                        adj[sp1, sp2] = 1
                        adj[sp2, sp1] = 1
            
            # 计算垂直方向的邻接关系
            for i in range(h-1):
                for j in range(w):
                    sp1 = sp_idx[i, j].item()
                    sp2 = sp_idx[i+1, j].item()
                    if sp1 != sp2:
                        adj[sp1, sp2] = 1
                        adj[sp2, sp1] = 1
            
            # 添加自环
            adj = adj + torch.eye(n_sp, device=sp_idx.device)
            
            # 归一化
            degree = adj.sum(dim=1).sqrt().unsqueeze(1)
            adj = adj / torch.matmul(degree, degree.t())
            
            all_adj.append(adj)
            
        return all_adj
    
    def graph_conv(self, sp_features, adj):
        """
        简化的图卷积操作
        
        Args:
            sp_features: 超像素特征 [N, C]
            adj: 邻接矩阵 [N, N]
            
        Returns:
            更新后的特征 [N, C]
        """
        return torch.matmul(adj, sp_features)
    
    def superpixel_infoNCE_loss(self, features1, features2, labels=None):
        """
        超像素级InfoNCE损失
        
        Args:
            features1: 第一视角特征 [N, D]
            features2: 第二视角特征 [N, D]
            labels: 超像素标签 [N]，可选
            
        Returns:
            对比损失
        """
        # 特征归一化
        features1 = F.normalize(features1, dim=1, p=2)
        features2 = F.normalize(features2, dim=1, p=2)
        
        # 计算相似度矩阵
        similarity = torch.matmul(features1, features2.t()) / self.temperature
        
        # 计算对角线位置的正样本损失
        n = features1.shape[0]
        positive_loss = torch.diagonal(similarity)
        
        # 创建对比学习掩码
        if labels is not None:
            # 使用标签创建掩码：相同类别为正样本(1)，不同类别为负样本(0)
            mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            # 移除对角线元素
            mask.fill_diagonal_(0)
        else:
            # 如果没有标签，则只有对角线位置是正样本
            mask = torch.zeros_like(similarity)
        
        # exp(-sim)用于所有样本对
        exp_sim = torch.exp(similarity)
        
        # 计算正样本对的损失
        if labels is not None and mask.sum() > 0:
            # 对每个样本，计算与所有其它同类样本的相似度
            pos_sim = similarity * mask
            # 将负无穷替换为0，以便在log计算中忽略这些位置
            pos_sim = pos_sim.masked_fill(mask == 0, float('-inf'))
            # 对每行取最大值
            pos_sim = torch.logsumexp(pos_sim, dim=1)
            
            # 计算负样本的损失
            neg_mask = 1 - mask
            neg_mask.fill_diagonal_(0)  # 排除对角线位置
            neg_sim = similarity * neg_mask
            neg_sim = torch.logsumexp(neg_sim, dim=1)
            
            # 组合正负样本损失
            contrastive_loss = -pos_sim + neg_sim
        else:
            # 标准InfoNCE损失：-log(exp(sim_pos) / sum(exp(sim)))
            denominator = torch.sum(exp_sim, dim=1)
            contrastive_loss = -positive_loss + torch.log(denominator)
        
        return contrastive_loss.mean()
    
    def get_superpixel_prototypes(self, features, labels, counts=None):
        """
        计算每个类别的超像素原型表示
        
        Args:
            features: 超像素特征 [N, D]
            labels: 超像素标签 [N]
            counts: 每个超像素包含的像素数 [N, 1]，用于加权
            
        Returns:
            类别原型特征 [C, D]
        """
        if counts is None:
            counts = torch.ones(features.shape[0], 1, device=features.device)
            
        prototypes = []
        
        for c in range(self.n_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                class_features = features[mask]
                class_counts = counts[mask]
                # 加权平均
                weighted_feat = class_features * class_counts
                prototype = weighted_feat.sum(dim=0) / class_counts.sum()
                prototypes.append(prototype)
            else:
                # 如果没有此类别的样本，使用零向量
                prototypes.append(torch.zeros(features.shape[1], device=features.device))
        
        return torch.stack(prototypes)
    
    def prototype_contrastive_loss(self, prototypes1, prototypes2):
        """
        原型级对比损失
        
        Args:
            prototypes1: 第一视角类别原型 [C, D]
            prototypes2: 第二视角类别原型 [C, D]
            
        Returns:
            原型对比损失
        """
        # 特征归一化
        prototypes1 = F.normalize(prototypes1, dim=1, p=2)
        prototypes2 = F.normalize(prototypes2, dim=1, p=2)
        
        # 计算相似度
        similarity = torch.matmul(prototypes1, prototypes2.t()) / self.temperature
        
        # 对角线位置为正样本
        positive = torch.diagonal(similarity)
        
        # 计算InfoNCE损失
        exp_sim = torch.exp(similarity)
        denominator = torch.sum(exp_sim, dim=1)
        
        loss = -positive + torch.log(denominator)
        return loss.mean()
    
    def generate_superpixel_labels(self, sp_features, tls_labels, superpixel_indices, threshold=0.5):
        """
        根据粗标注生成超像素级标签
        
        Args:
            sp_features: 超像素特征 [N, D]
            tls_labels: TLS区域标签 [H, W]
            superpixel_indices: 超像素索引 [H, W]
            threshold: 标签分配阈值
            
        Returns:
            超像素标签 [N]
        """
        n_sp = sp_features.shape[0]
        sp_labels = torch.zeros(n_sp, device=sp_features.device, dtype=torch.long)
        
        # 展平标签和索引
        tls_labels_flat = tls_labels.reshape(-1)
        sp_indices_flat = superpixel_indices.reshape(-1)
        
        # 计算每个超像素区域中TLS像素的比例
        for i in range(n_sp):
            mask = (sp_indices_flat == i)
            if mask.sum() > 0:
                tls_pixels = tls_labels_flat[mask].float().mean()
                # 超过阈值的超像素被标记为TLS区域(1)，否则为背景(0)
                sp_labels[i] = 1 if tls_pixels > threshold else 0
        
        return sp_labels
    
    def forward(self, features, superpixel_indices, tls_labels=None):
        """
        前向传播
        
        Args:
            features: 像素级特征 [B, C, H, W]
            superpixel_indices: 超像素索引 [B, H, W]
            tls_labels: TLS区域标签 [B, H, W]，可选
            
        Returns:
            超像素对比损失
        """
        batch_size = features.shape[0]
        total_loss = 0.0
        
        # 聚合超像素特征
        all_sp_features, all_sp_counts = self.aggregate_superpixel_features(features, superpixel_indices)
        
        # 构建邻接矩阵
        all_adj = self.build_adjacency_matrix(superpixel_indices)
        
        for b in range(batch_size):
            sp_features = all_sp_features[b]  # [N, C]
            adj = all_adj[b]  # [N, N]
            
            # 应用图卷积
            sp_features_conv = self.graph_conv(sp_features, adj)
            
            # 通过编码器和双分支投影头
            h = self.encoder(sp_features_conv)
            z1 = self.projector1(h)
            z2 = self.projector2(h)
            
            # 如果有TLS标签，生成超像素标签
            if tls_labels is not None:
                sp_labels = self.generate_superpixel_labels(
                    sp_features, tls_labels[b], superpixel_indices[b])
                
                # 计算类别原型
                prototypes1 = self.get_superpixel_prototypes(z1, sp_labels, all_sp_counts[b])
                prototypes2 = self.get_superpixel_prototypes(z2, sp_labels, all_sp_counts[b])
                
                # 计算原型对比损失
                proto_loss = self.prototype_contrastive_loss(prototypes1, prototypes2)
                
                # 计算超像素对比损失
                sp_loss = self.superpixel_infoNCE_loss(z1, z2, sp_labels)
                
                loss = sp_loss + proto_loss
            else:
                # 无标签情况下，只使用基本对比损失
                loss = self.superpixel_infoNCE_loss(z1, z2)
            
            total_loss += loss
            
        return total_loss / batch_size

    def extract_superpixel_features(self, features, superpixel_indices):
        """
        提取最终的超像素特征表示(用于下游任务)
        
        Args:
            features: 像素级特征 [B, C, H, W]
            superpixel_indices: 超像素索引 [B, H, W]
            
        Returns:
            超像素特征 [B, N, D]，超像素数量，特征维度
        """
        batch_size = features.shape[0]
        all_final_features = []
        
        # 聚合超像素特征
        all_sp_features, _ = self.aggregate_superpixel_features(features, superpixel_indices)
        
        # 构建邻接矩阵
        all_adj = self.build_adjacency_matrix(superpixel_indices)
        
        for b in range(batch_size):
            sp_features = all_sp_features[b]  # [N, C]
            adj = all_adj[b]  # [N, N]
            
            # 应用图卷积
            sp_features_conv = self.graph_conv(sp_features, adj)
            
            # 通过编码器和第一个投影头
            h = self.encoder(sp_features_conv)
            z = self.projector1(h)
            
            all_final_features.append(z)
            
        return all_final_features
