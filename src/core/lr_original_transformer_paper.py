import math
import torch.optim as optim

class LinearOriginalTransformerScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, fea_dim, max_iters):
        # note: save the params first, then call super.__init__
        # otherwise the paramss will not be saved.
        self.warmup = warmup
        self.max_iters = max_iters
        self.fea_dim = fea_dim
        super().__init__(optimizer)

    def get_lr(self):
        current_step = max(self.last_epoch + 1, 1)  # Handle division by zero
        return [
            (self.fea_dim ** -0.5) *
            min(current_step ** -0.5, current_step * (self.warmup ** -1.5))
            for _ in self.base_lrs
        ]


if __name__ == '__main__':

    import torch
    import torch.nn as nn

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        
        def forward(self, x):
            return self.fc(x)

    model = SimpleNet()

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=1e-3,           # Default is 0.001
        betas=(0.9, 0.999), # Coefficients for running averages
        weight_decay=0.0001
    )

    scheduler = LinearOriginalTransformerScheduler(optimizer, 1000, 256, 100)

    for epoch in range(10):
        for batch_idx, batch in enumerate(range(10)):
            # Forward pass
            # inputs = torch.randn(5, 10)  # Batch of 5 samples
            # outputs = model(inputs)
            # targets = torch.randint(0, 2, (5,))  # Fake targets
            # loss = nn.CrossEntropyLoss()(outputs, targets)
            
            optimizer.step()       
            scheduler.step()
            print(scheduler.get_lr())

