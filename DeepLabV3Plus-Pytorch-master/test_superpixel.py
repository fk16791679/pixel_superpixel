import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.superpixel_utils import generate_superpixels

def test_superpixel_segmentation(image_path, n_segments=100, compactness=10.0):
    # 读取图像
    image = np.array(Image.open(image_path))
    
    # 生成超像素分割
    segments = generate_superpixels(image, n_segments=n_segments, compactness=compactness)
    
    # 使用不同的颜色显示不同的超像素区域
    from skimage.color import label2rgb
    segmentation_overlay = label2rgb(segments, image, kind='avg', bg_label=-1)
    
    return segmentation_overlay, len(np.unique(segments))

def test_multiple_parameters(image_path):
    # 定义要测试的参数组合
    n_segments_list = [20, 100, 200, 500]
    compactness_list = [1, 5, 10, 20]
    
    # 读取原始图像
    image = np.array(Image.open(image_path))
    
    # 创建子图网格
    rows = len(compactness_list) + 1
    cols = len(n_segments_list) + 1
    fig = plt.figure(figsize=(20, 20))
    
    # 显示原始图像在左上角
    ax = plt.subplot(rows, cols, 1)
    ax.imshow(image)
    ax.set_title('原始图像')
    ax.axis('off')
    
    # 生成并显示不同参数组合的结果
    for i, compactness in enumerate(compactness_list, 1):
        for j, n_segments in enumerate(n_segments_list, 1):
            # 生成分割结果
            segmentation_overlay, actual_segments = test_superpixel_segmentation(
                image_path,
                n_segments=n_segments,
                compactness=compactness
            )
            
            # 在网格中显示结果
            ax = plt.subplot(rows, cols, (i * cols) + j)
            ax.imshow(segmentation_overlay)
            ax.set_title(f'n_segments={n_segments}\ncompactness={compactness}\n实际分割数={actual_segments}')
            ax.axis('off')
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存结果
    output_path = image_path.rsplit('.', 1)[0] + '_parameter_comparison.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f'参数对比结果已保存到: {output_path}')

if __name__ == '__main__':
    # 替换为你的测试图像路径
    image_path = r"C:\Users\13409\Desktop\1500647_50_32.png"
    
    # 测试多组参数
    test_multiple_parameters(image_path)