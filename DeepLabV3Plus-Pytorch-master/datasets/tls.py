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
        mask_dir = os.path.join(self.root, split, 'binary_mask')
        
        # 获取所有图片文件
        self.images = sorted([os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith('.png')])
        self.masks = sorted([os.path.join(mask_dir, x) for x in os.listdir(mask_dir) if x.endswith('.png')])
        
        assert len(self.images) == len(self.masks), "图像和标签数量不匹配"
        
        # 打印一些信息用于调试
        print(f"数据集{split}: 找到{len(self.images)}个图像和掩码对")

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        
        # 读取掩码并转换为灰度图
        target = Image.open(self.masks[index]).convert('L')  # 转为灰度图
        
        # 将掩码转换为numpy数组，以便处理
        target_np = np.array(target)
        
        # 检查掩码值并转换为二值掩码（0=背景，1=前景）
        # 这里假设红色区域的像素值较高
        if target_np.max() > 1:  # 如果掩码不是0-1值
            print("max",target_np.max())
            # 使用阈值将其转换为二值掩码
            threshold = 127  # 中间值，可能需要根据实际情况调整
            binary_target = (target_np > threshold).astype(np.int64)
            target = Image.fromarray(binary_target)
        
        # 应用变换
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        # 确保target是LongTensor类型，这是PyTorch分割任务的要求
        if hasattr(target, 'dtype') and not torch.is_floating_point(target) and not torch.is_complex(target):
            if target.dtype != torch.int64:
                target = target.long()
        
        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def decode_target(mask):
        """将mask解码为RGB图像"""
        # 更改颜色方案，使其与你提到的红色TLS更一致
        colors = np.array([[0, 0, 0], [255, 0, 0]])  # 背景黑色，TLS区域红色
        return colors[mask]
