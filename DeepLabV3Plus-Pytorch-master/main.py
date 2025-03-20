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
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
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
    parser.add_argument("--total_itrs", type=int, default=30e2,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
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
    # 删除 gpu_id 参数
    parser.add_argument("--gpu_ids", type=str, default='0,1,2,3,4,5,6,7',
                        help="GPU IDs (default: 0,1,2,3)")
    return parser

def get_dataset(opts):
        # 添加图像大小检查和调整
        train_transform = et.ExtCompose([
            et.ExtResize(size=(opts.crop_size, opts.crop_size), interpolation=InterpolationMode.BILINEAR),  # Updated interpolation
            et.ExtRandomHorizontalFlip(),
            et.ExtRandomVerticalFlip(),
            et.ExtRandomRotation(90),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtResize(size=(opts.crop_size, opts.crop_size), interpolation=InterpolationMode.BILINEAR),  # Updated interpolation
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


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    opts.num_classes = 2
    writer = SummaryWriter(f'runs_pixel/tls_{opts.model}')
    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    # 改为使用 gpu_ids
    # 修改GPU设置部分
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只使用第一个GPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using single GPU")
    num_gpus = len(opts.gpu_ids.split(','))
    print("Number of GPUs: %d" % num_gpus)
    opts.batch_size = opts.batch_size * num_gpus
    opts.val_batch_size = opts.val_batch_size * num_gpus
    opts.lr = opts.lr * num_gpus
    # 调整batch size和学习率

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)


    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, 
        num_workers=min(8, 2),  # 限制worker数量
        drop_last=True)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, 
        num_workers=min(8, 2))
    iters_per_epoch = len(train_loader)
    total_epochs = opts.total_itrs // iters_per_epoch + 1
    print(f"训练集大小: {len(train_dst)}, 每个epoch的迭代次数: {iters_per_epoch}")
    print(f"计划训练epoch数: {total_epochs}")
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    if opts.use_pixel_contrast:
        pixel_contrast = PixelContrastiveLearning(temperature=opts.temperature).to(device)
    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        
    # 移至设备前先迁移模型
    model.to(device)

    # 只选择一种并行方式
    if torch.cuda.device_count() > 1:
        print("使用 %d 个 GPU!" % torch.cuda.device_count())
        model = nn.DataParallel(model)
    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    interval_ce_loss =0
    interval_pixel_loss =0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        epoch_loss = 0
        for (images, labels) in train_loader:
            cur_itrs += 1
            cur_epochs = (cur_itrs - 1) // iters_per_epoch + 1  
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            # loss = criterion(outputs, labels)
            ce_loss = criterion(outputs, labels)
            if opts.use_pixel_contrast:
            # 获取特征图（假设模型有返回特征图的接口）
                features = model.module.backbone(images)['out']
                pixel_loss = pixel_contrast(features, labels)
                current_progress = cur_itrs / opts.total_itrs
                pixel_contrast_weight = opts.pixel_contrast_weight * (1 - current_progress)  # 随训练进度逐渐减小权重
                loss = ce_loss + pixel_contrast_weight * pixel_loss
                
                # 记录对比学习损失
                writer.add_scalar('Loss/pixel_contrast', pixel_loss.item(), cur_itrs)
            else:
                loss = ce_loss
                
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            if opts.use_pixel_contrast:
                np_ce_loss = ce_loss.detach().cpu().numpy()
                np_pixel_loss = pixel_loss.detach().cpu().numpy()
                interval_ce_loss += np_ce_loss
                interval_pixel_loss += np_pixel_loss
            interval_loss += np_loss

            writer.add_scalar('Loss/train', np_loss, cur_itrs)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], cur_itrs)
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                if opts.use_pixel_contrast:
                    interval_ce_loss = interval_ce_loss / 10
                    interval_pixel_loss = interval_pixel_loss / 10
                    print("Epoch %d, Itrs %d/%d, Total Loss=%f, CE Loss=%f, Pixel Loss=%f" %
                        (cur_epochs, cur_itrs, opts.total_itrs, interval_loss, 
                        interval_ce_loss, interval_pixel_loss))
                    interval_ce_loss = 0.0
                    interval_pixel_loss = 0.0
                else:
                    print("Epoch %d, Itrs %d/%d, Total Loss=%f" %
                        (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0


            val_interval = max(opts.val_interval, int(0.1 * iters_per_epoch))  # 至少间隔0.1个epoch
            if (cur_itrs) % val_interval == 0:

                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                writer.add_scalar('Metrics/val_acc', val_score['Overall Acc'], cur_itrs)
                writer.add_scalar('Metrics/mean_iou', val_score['Mean IoU'], cur_itrs)
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                writer.close()
                return
        epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train_epoch', epoch_loss, cur_epochs)


if __name__ == '__main__':
    main()
