
import math
import torch.optim as optim

class LinearWarmupInvSqrtDecayScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, fea_dim, max_iters):
        # note: save the params first, then call super.__init__
        # otherwise the paramss will not be saved.
        self.warmup = warmup
        self.max_iters = max_iters
        self.fea_dim = fea_dim
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch+1) # avoid 0
        return [base_lr * lr_factor for base_lr in self.base_lrs]
    
    def get_lr_factor(self, epoch):
        if epoch < self.warmup:
            return epoch / max(1, self.warmup)
        else:
            return (self.warmup ** 0.5) / (epoch ** 0.5)
            
        