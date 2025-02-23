import math
import torch.optim as optim

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    From https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """
    def __init__(self, optimizer, warmup, max_iters):
        # note: save the params first, then call super.__init__
        # otherwise the paramss will not be saved.
        self.warmup = warmup
        self.max_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]
    
    # def get_lr_factor(self, epoch):
    #     if epoch < self.warmup: # linear warmup
    #         return epoch / self.warmup
    #     else:
    #         # consine decay
    #         progress = (epoch - self.warmup) / (self.max_iters - self.warmup)
    #         return 0.5 * (1 + math.cos(math.pi * progress))

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / self.max_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor        