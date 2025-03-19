import numpy as np
from skimage.segmentation import slic
import torch

def generate_superpixels(image, n_segments=100, compactness=10.0):
    """
    使用SLIC算法生成超像素分割
    
    Args:
        image: numpy数组，形状为[H, W, C]的RGB图像
        n_segments: 期望的超像素数量
        compactness: SLIC算法的紧密度参数
        
    Returns:
        超像素索引图，形状为[H, W]的numpy数组
    """
    # 确保图像是uint8类型
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # 应用SLIC算法
    segments = slic(image, n_segments=n_segments, compactness=compactness,
                   start_label=0, convert2lab=True)
    
    return segments

def batch_generate_superpixels(images, n_segments=100, compactness=10.0):
    """
    对批量图像生成超像素分割
    
    Args:
        images: Tensor，形状为[B, C, H, W]的批量图像
        n_segments: 期望的超像素数量
        compactness: SLIC算法的紧密度参数
        
    Returns:
        超像素索引图，形状为[B, H, W]的Tensor
    """
    batch_size = images.shape[0]
    device = images.device
    
    # 将图像转换为numpy数组并调整通道顺序
    images_np = images.cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, C]
    
    # 对每个图像生成超像素
    segments_list = []
    for i in range(batch_size):
        segments = generate_superpixels(images_np[i], n_segments, compactness)
        segments_list.append(segments)
    
    # 转换为Tensor并移回原设备
    segments_tensor = torch.from_numpy(np.stack(segments_list)).to(device)
    
    return segments_tensor