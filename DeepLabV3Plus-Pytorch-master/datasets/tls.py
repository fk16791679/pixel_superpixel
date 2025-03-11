import os
import torch.utils.data as data
import numpy as np
from PIL import Image

class TLSSegmentation(data.Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        
        # 直接指向TLS数据目录
        image_dir = os.path.join(self.root, split, 'images')
        mask_dir = os.path.join(self.root, split, 'masks')
        
        # 获取所有图片文件
        self.images = sorted([os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith('.png')])
        self.masks = sorted([os.path.join(mask_dir, x) for x in os.listdir(mask_dir) if x.endswith('.png')])
        
        assert len(self.images) == len(self.masks), "图像和标签数量不匹配"

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index]).convert('L')  # 转为灰度图
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def decode_target(mask):
        """将mask解码为RGB图像"""
        colors = np.array([[0, 0, 0], [255, 255, 255]])  # 背景黑色，TLS区域白色
        return colors[mask]