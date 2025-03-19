from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from datasets.tls import TLSSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from torchvision.transforms import InterpolationMode
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from utils.pixel_loss import PixelContrastiveLearning
from utils.superpixel_loss import SuperpixelContrastiveLearning
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
torch.backends.cudnn.enabled = False

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/home/sdd/fanke/two/data/',
                        help="path to Dataset")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")
    parser.add_argument("--use_pixel_contrast", action='store_true', default=False,
                        help="use pixel-level contrastive learning")
    parser.add_argument("--pixel_contrast_weight", type=float, default=0.1,
                        help="weight for pixel-level contrastive loss")
    parser.add_argument("--use_superpixel_contrast", action='store_true', default=False,
                        help="use superpixel-level contrastive learning")
    parser.add_argument("--superpixel_contrast_weight", type=float, default=0.1,
                        help="weight for superpixel-level contrastive loss")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="temperature for contrastive loss")
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=True,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=10000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=32,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=32,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy', help="loss type (default: False)")

    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--dataset", type=str, default='tls',
                        help="dataset name")
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    
    # DDP相关参数
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed training")
    parser.add_argument("--use_amp", action='store_true', default=False,
                        help="Use automatic mixed precision")
    
    # 保留gpu_ids参数以兼容旧代码
    parser.add_argument("--gpu_ids", type=str, default='0,1,2,3,4,5,6,7',
                        help="GPU IDs (default: 0,1,2,3)")
    return parser

def get_dataset(opts):
        # 添加图像大小检查和调整
        train_transform = et.ExtCompose([
            et.ExtResize(size=(opts.crop_size, opts.crop_size), interpolation=InterpolationMode.BILINEAR),
            et.ExtRandomHorizontalFlip(),
            et.ExtRandomVerticalFlip(),
            et.ExtRandomRotation(90),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtResize(size=(opts.crop_size, opts.crop_size), interpolation=InterpolationMode.BILINEAR),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
        ])

        train_dst = TLSSegmentation(root=opts.data_root,
                                  split='train',
                                  transform=train_transform)
        val_dst = TLSSegmentation(root=opts.data_root,
                                split='val',
                                transform=val_transform)
        return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None, rank=0):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results and rank == 0:  # 仅在主进程保存结果
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(loader)):
            images, labels = data[0], data[1]
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids and rank == 0:  # 仅在主进程获取样本
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results and rank == 0:  # 仅在主进程保存结果
                # ... [保存结果的代码] ...
                pass

        # 汇总所有进程的评估结果
        if dist.is_initialized():
            # 检查metrics对象结构，同步正确的属性
            # 方法1：直接同步混淆矩阵（假设metrics有_confusion_matrix属性）
            if hasattr(metrics, 'confusion_matrix'):
                conf_matrix = torch.tensor(metrics.confusion_matrix, device=device)
                dist.all_reduce(conf_matrix)
                metrics.confusion_matrix = conf_matrix.cpu().numpy() / dist.get_world_size()
            elif hasattr(metrics, '_confusion_matrix'):
                conf_matrix = torch.tensor(metrics._confusion_matrix, device=device)
                dist.all_reduce(conf_matrix)
                metrics._confusion_matrix = conf_matrix.cpu().numpy() / dist.get_world_size()
            # 方法2：如果没有明确的混淆矩阵，只同步最终结果
            else:
                # 在这里同步其他可能的指标属性
                # 由于不确定具体的属性名，可以先获取结果再同步
                result = metrics.get_results()
                for k in result.keys():
                    if isinstance(result[k], (int, float)):
                        tensor = torch.tensor(result[k], device=device)
                        dist.all_reduce(tensor)
                        result[k] = tensor.item() / dist.get_world_size()

        score = metrics.get_results()
    return score, ret_samples


def main():
    parser = get_argparser()
    opts = parser.parse_args()
    opts.num_classes = 2

    # 初始化分布式训练环境
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 使用torchrun或torch.distributed.launch启动的进程
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        opts.local_rank = local_rank
    else:
        # 单进程环境
        rank = 0
        world_size = 1
        local_rank = 0
    
    # 设置当前设备
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    
    # 初始化进程组
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://',
                               world_size=world_size, rank=rank)
        print(f"Process {rank}/{world_size} using GPU: {local_rank}")

    # 仅在主进程上设置SummaryWriter
    writer = SummaryWriter(f'runs_pixel/tls_{opts.model}') if rank == 0 else None
    
    # 可视化工具仅在主进程上设置
    vis = Visualizer(port=opts.vis_port, env=opts.vis_env) if opts.enable_vis and rank == 0 else None
    if vis is not None:
        vis.vis_table("Options", vars(opts))

    # 设置随机种子
    torch.manual_seed(opts.random_seed + rank)
    np.random.seed(opts.random_seed + rank)
    random.seed(opts.random_seed + rank)
    
    # 保存原始批次大小
    original_batch_size = opts.batch_size
    original_val_batch_size = opts.val_batch_size
    
    # 根据进程数调整批次大小，确保至少为1
    if world_size > 1:
        # 使用整除，并确保结果至少为1
        opts.batch_size = max(1, original_batch_size // world_size)
        opts.val_batch_size = max(1, original_val_batch_size // world_size)
    
    # 确保批次大小为正整数
    opts.batch_size = max(1, opts.batch_size)
    opts.val_batch_size = max(1, opts.val_batch_size)
    
    if rank == 0:
        print(f"原始批次大小: train={original_batch_size}, val={original_val_batch_size}")
        print(f"每个GPU使用的批次大小: train={opts.batch_size}, val={opts.val_batch_size}")
        print(f"Using {world_size} GPUs, learning rate={opts.lr}")

    # 获取数据集
    train_dst, val_dst = get_dataset(opts)
    
    # 为训练和验证集设置分布式采样器
    train_sampler = DistributedSampler(train_dst) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dst, shuffle=False) if world_size > 1 else None

    # 创建数据加载器
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,  # 增加workers数量提高数据加载效率
        pin_memory=True,  # 使用pin_memory加快数据传输
        drop_last=True)
    
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size,
        shuffle=False,  # 在验证时不需要shuffle
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True)
    
    # 计算训练迭代次数
    iters_per_epoch = len(train_loader)
    total_epochs = opts.total_itrs // iters_per_epoch + 1
    
    if rank == 0:
        print(f"训练集大小: {len(train_dst)}, 每个epoch的迭代次数: {iters_per_epoch}")
        print(f"计划训练epoch数: {total_epochs}")
        print(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)}")

    # 设置模型
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # 将模型移至设备
    model.to(device)
    
    # 使用DDP包装模型
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # 设置评估指标
    metrics = StreamSegMetrics(opts.num_classes)

    # 设置优化器
    # 注意：在DDP中，优化器应该只操作模块参数
    if hasattr(model, 'module'):
        backbone_params = model.module.backbone.parameters()
        classifier_params = model.module.classifier.parameters()
    else:
        backbone_params = model.backbone.parameters()
        classifier_params = model.classifier.parameters()
        
    optimizer = torch.optim.SGD([
        {'params': backbone_params, 'lr': 0.1 * opts.lr},
        {'params': classifier_params, 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    
    # 设置学习率调度器
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # 设置损失函数
    if opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        
    # 设置像素对比损失和超像素对比损失
    if opts.use_pixel_contrast:
        pixel_contrast = PixelContrastiveLearning(temperature=opts.temperature).to(device)
    if opts.use_superpixel_contrast:
        superpixel_contrast = SuperpixelContrastiveLearning(temperature=opts.temperature).to(device)
    # 设置自动混合精度
    scaler = GradScaler() if opts.use_amp else None
    
    # 模型保存函数
    def save_ckpt(path):
        """保存当前模型"""
        # 在分布式训练中，只在主进程保存模型
        if rank == 0:
            if hasattr(model, 'module'):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
                
            torch.save({
                "cur_itrs": cur_itrs,
                "model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_score": best_score,
            }, path)
            print(f"Model saved as {path}")

    # 创建检查点目录
    if rank == 0:
        utils.mkdir('checkpoints')
        
    # 恢复检查点
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=device)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint["model_state"])
            
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint["best_score"]
            
            if rank == 0:
                print("继续训练，从迭代 %d 开始" % cur_itrs)

    # 设置可视化样本ID
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples, 
                                     np.int32) if opts.enable_vis else None
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 如果仅测试，则运行验证并退出
    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, 
            ret_samples_ids=vis_sample_id, rank=rank)
        
        if rank == 0:
            print(metrics.to_str(val_score))
        return

    # 训练循环
    interval_loss = 0
    interval_ce_loss = 0
    interval_pixel_loss = 0
    interval_superpixel_loss = 0

    while True:
        # 训练阶段
        model.train()
        epoch_loss = 0
        
        # 在每个epoch开始时设置sampler的epoch
        if train_sampler is not None:
            train_sampler.set_epoch(cur_epochs)
            
        # 使用tqdm只在rank=0时显示进度条
        train_iter = tqdm(train_loader, disable=rank!=0)
        
        for (images, labels, superpixel_indices) in train_iter:
            cur_itrs += 1
            cur_epochs = (cur_itrs - 1) // iters_per_epoch + 1
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            superpixel_indices = superpixel_indices.to(device, dtype=torch.long)

            optimizer.zero_grad()
            
            # 使用自动混合精度
            if opts.use_amp:
                with autocast():
                    outputs = model(images)
                    ce_loss = criterion(outputs, labels)
                    
                    # 计算特征图
                    if opts.use_pixel_contrast or opts.use_superpixel_contrast:
                        if hasattr(model, 'module'):
                            features = model.module.backbone(images)['out']
                        else:
                            features = model.backbone(images)['out']
                    
                    # 计算像素对比损失
                    if opts.use_pixel_contrast:
                        pixel_loss = pixel_contrast(features, labels)
                    else:
                        pixel_loss = torch.tensor(0.0, device=device)
                    
                    # 计算超像素对比损失
                    if opts.use_superpixel_contrast:
                        superpixel_loss = superpixel_contrast(features, labels, superpixel_indices)
                    else:
                        superpixel_loss = torch.tensor(0.0, device=device)
                    
                    # 组合所有损失
                    loss = ce_loss + \
                           opts.pixel_contrast_weight * pixel_loss + \
                           opts.superpixel_contrast_weight * superpixel_loss
                        
                # 使用scaler进行反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)  
                scaler.update()
                scheduler.step()
            else:
                # 不使用混合精度
                outputs = model(images)
                ce_loss = criterion(outputs, labels)
                
                # 计算特征图
                if opts.use_pixel_contrast or opts.use_superpixel_contrast:
                    if hasattr(model, 'module'):
                        features = model.module.backbone(images)['out']
                    else:
                        features = model.backbone(images)['out']
                
                # 计算像素对比损失
                if opts.use_pixel_contrast:
                    pixel_loss = pixel_contrast(features, labels)
                else:
                    pixel_loss = torch.tensor(0.0, device=device)
                
                # 计算超像素对比损失
                if opts.use_superpixel_contrast:
                    superpixel_loss = superpixel_contrast(features, labels)
                else:
                    superpixel_loss = torch.tensor(0.0, device=device)
                
                # 组合所有损失
                loss = ce_loss + \
                       opts.pixel_contrast_weight * pixel_loss + \
                       opts.superpixel_contrast_weight * superpixel_loss
                    
                loss.backward()
                optimizer.step()
                # 更新学习率
                scheduler.step()
            # 收集损失值
            np_loss = loss.detach().cpu().numpy()
            epoch_loss += np_loss
            np_ce_loss = ce_loss.detach().cpu().numpy()
            np_pixel_loss = pixel_loss.detach().cpu().numpy()
            np_superpixel_loss = superpixel_loss.detach().cpu().numpy()
            interval_ce_loss += np_ce_loss
            interval_pixel_loss += np_pixel_loss
            interval_superpixel_loss += np_superpixel_loss
            interval_loss += np_loss

            # 记录损失和学习率
            if rank == 0:
                writer.add_scalar('Loss/train', np_loss, cur_itrs)
                writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], cur_itrs)
                if opts.use_pixel_contrast:
                    writer.add_scalar('Loss/pixel_contrast', np_pixel_loss, cur_itrs)
                    
                if vis is not None:
                    vis.vis_scalar('Loss', cur_itrs, np_loss)

            # 打印训练信息
            if (cur_itrs) % opts.print_interval == 0 and rank == 0:
                interval_loss = interval_loss / opts.print_interval
                interval_ce_loss = interval_ce_loss / opts.print_interval
                interval_pixel_loss = interval_pixel_loss / opts.print_interval
                interval_superpixel_loss = interval_superpixel_loss / opts.print_interval
                print("Epoch %d, Itrs %d/%d, Total Loss=%f, CE Loss=%f, Pixel Loss=%f, Superpixel Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss, 
                       interval_ce_loss, interval_pixel_loss, interval_superpixel_loss))
                interval_ce_loss = 0.0
                interval_pixel_loss = 0.0
                interval_superpixel_loss = 0.0
                interval_loss = 0.0

            # 验证模型
            if (cur_itrs) % opts.val_interval == 0:
                # 保存检查点
                save_ckpt('checkpoints_pixel/latest_%s_%s_os%d.pth' %
                         (opts.model, opts.dataset, opts.output_stride))
                
                if rank == 0:
                    print("validation...")
                    
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id, rank=rank)
                
                if rank == 0:
                    print(metrics.to_str(val_score))
                    writer.add_scalar('Metrics/val_acc', val_score['Overall Acc'], cur_itrs)
                    writer.add_scalar('Metrics/mean_iou', val_score['Mean IoU'], cur_itrs)
                    
                    # 保存最佳模型
                    if val_score['Mean IoU'] > best_score:
                        best_score = val_score['Mean IoU']
                        save_ckpt('checkpoints_pixel/best_%s_%s_os%d.pth' %
                                 (opts.model, opts.dataset, opts.output_stride))

                    # 可视化验证结果
                    if vis is not None:
                        vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                        vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                        vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                        for k, (img, target, lbl) in enumerate(ret_samples):
                            img = (denorm(img) * 255).astype(np.uint8)
                            target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                            lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                            concat_img = np.concatenate((img, target, lbl), axis=2)
                            vis.vis_image('Sample %d' % k, concat_img)
                            
                # 同步所有进程后继续训练
                if world_size > 1:
                    dist.barrier()
                    
                model.train()
                


            # 检查是否完成训练
            if cur_itrs >= opts.total_itrs:
                if rank == 0:
                    writer.close()
                # 同步所有进程后退出
                if world_size > 1:
                    dist.destroy_process_group()
                return
                
        # 计算每个epoch的平均损失
        if rank == 0:
            epoch_loss = epoch_loss / len(train_loader)
            writer.add_scalar('Loss/train_epoch', epoch_loss, cur_epochs)

if __name__ == '__main__':
    # 如果使用torchrun或distributed.launch，main函数会被多次调用
    main()
