from torch.optim.lr_scheduler import _LRScheduler, StepLR

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-5):
        self.power = power
        self.max_iters = max_iters
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # 使用余弦退火策略结合多项式衰减
        factor = (1 - self.last_epoch/self.max_iters)**self.power
        cosine_factor = 0.5 * (1 + math.cos(math.pi * self.last_epoch/self.max_iters))
        return [max(base_lr * factor * cosine_factor, self.min_lr)
                for base_lr in self.base_lrs]