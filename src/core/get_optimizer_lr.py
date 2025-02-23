import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_optimizer_with_lr(name):
    match name:
        case "cosine_warmup": 
            return optimizer_with_cosine_warmup
        case "linear_warmup_inv_sqrt_decay":
            return optimizer_with_linear_inv_sqrt_decay
        case "linear_const_lr":
            return optimizer_with_const_lr
        case "reduce_lr_on_plateau":
            return optimizer_with_reduce_lr_on_plateau
        case "bert_fine_tuning":
            return bert_fine_tuning
        case "original_transformer_lr":
            return optimizer_with_original_transformer_lr
        
def bert_fine_tuning(parameters, **kwargs):
    # huggingface
    from transformers import AdamW, get_linear_schedule_with_warmup

    optimizer = AdamW(params=parameters, lr=2e-5, correct_bias=False)
    total_steps = kwargs['total_steps'] # len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1*total_steps,
        num_training_steps=total_steps
    )
    return [optimizer], [scheduler]

# todo:
def optimizer_with_reduce_lr_on_plateau(parameters, **kwargs):
    optimizer = optim.Adam(params=parameters, lr=kwargs["lr"])    
    scheduler = {
        'scheduler': ReduceLROnPlateau(
            optimizer,
            mode='min',         # For val_loss, use 'min'; for accuracy, use 'max'
            factor=0.1,         # Factor by which LR is reduced
            patience=3,         # Number of epochs with no improvement
            verbose=True
        ),
        'interval': 'epoch',    # Scheduler step runs after each epoch
        'monitor': 'val_loss'    # Metric to monitor
    }

    return [optimizer], [scheduler]

def optimizer_with_const_lr(parameters, **kwargs):
    optimizer = optim.Adam(params=parameters, lr=kwargs["lr"])    

    # Add default scheduler
    scheduler = {
        'scheduler': optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,  # Adjust based on your needs
            gamma=0.1      # LR multiplied by 0.1 every 30 epochs
        ),
        'interval': 'epoch',  # 'step' for per-batch updates
        'frequency': 1
    }

    return [optimizer], [scheduler]

def optimizer_with_cosine_warmup(parameters, **kwargs):
    from core.lr_cosine_warmup import CosineWarmupScheduler

    optimizer = optim.Adam(params=parameters, lr=kwargs["lr"])

    lr_scheduler = CosineWarmupScheduler(optimizer,
                                         warmup=kwargs["warmup"],
                                         max_iters=kwargs["max_iters"])
        
    return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

def optimizer_with_linear_inv_sqrt_decay(parameters, **kwargs):
    import torch.optim as optim
    from core.lr_linear_warmup_inv_sqrt_decay import LinearWarmupInvSqrtDecayScheduler

    optimizer = optim.Adam(params=parameters, lr=kwargs["lr"])

    lr_scheduler = LinearWarmupInvSqrtDecayScheduler(optimizer,
                                                     warmup=kwargs["warmup"],
                                                     max_iters=kwargs["max_iters"],
                                                     fea_dim=kwargs["fea_dim"])    
    return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

def optimizer_with_original_transformer_lr(parameters, **kwargs):
    import torch.optim as optim
    from core.lr_original_transformer_paper import LinearOriginalTransformerScheduler

    optimizer = optim.Adam(params=parameters, lr=kwargs["lr"])

    lr_scheduler = LinearOriginalTransformerScheduler(optimizer,
                                                     warmup=kwargs["warmup"],
                                                     max_iters=kwargs["max_iters"],
                                                     fea_dim=kwargs["fea_dim"])    
    return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]