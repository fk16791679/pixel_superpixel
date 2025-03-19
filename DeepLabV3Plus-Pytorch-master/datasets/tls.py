
from utils.superpixel_utils import generate_superpixels
import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import torch
class TLSSegmentation(data.Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        
        # 直接指向TLS数据目录
        image_dir = os.path.join(self.root, split, 'images')
        mask_dir = os.path.join(self.root, split, 'binary_mask')
        
        # 获取所有图片文件
        self.images = sorted([os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith('.png')])
        self.masks = sorted([os.path.join(mask_dir, x) for x in os.listdir(mask_dir) if x.endswith('.png')])
        
        assert len(self.images) == len(self.masks), "图像和标签数量不匹配"
        
        # 打印一些信息用于调试
        print(f"数据集{split}: 找到{len(self.images)}个图像和掩码对")

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        
        # 生成超像素索引
        img_np = np.array(img)
        superpixel_index = generate_superpixels(img_np)
        
        # 读取掩码并转换为灰度图
        target = Image.open(self.masks[index]).convert('L')
        
        # 将掩码转换为numpy数组，并归一化到0-1
        target_np = np.array(target)
        binary_target = (target_np > 127).astype(np.uint8)  # 将255值转换为1
        
        # 转换为PIL图像
        target = Image.fromarray(binary_target)
        
        # 应用变换
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        # 确保target是LongTensor类型
        if isinstance(target, torch.Tensor):
            target = target.long()
        
        # 将超像素索引转换为tensor
        superpixel_index = torch.from_numpy(superpixel_index).long()
        
        return img, target, superpixel_index

    def __len__(self):
        return len(self.images)

    @staticmethod
    def decode_target(mask):
        """将mask解码为RGB图像"""
        # 更改颜色方案，使其与你提到的红色TLS更一致
        colors = np.array([[0, 0, 0], [255, 0, 0]])  # 背景黑色，TLS区域红色
        return colors[mask]
