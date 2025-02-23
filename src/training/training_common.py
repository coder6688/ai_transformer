import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from config.env import CHECKPOINT_PATH, device, get_device
from core.task_trainer import TaskTrainer
from visualization.plot_attention_mat import plot_attention_mat


def training(train_loader, val_loader, test_loader, **kwargs):
    root_dir = os.path.join(CHECKPOINT_PATH, kwargs["task_name"])
    os.makedirs(root_dir, exist_ok=True)

    trainer = pl.Trainer(default_root_dir=root_dir,
                         logger=[CSVLogger(root_dir), TensorBoardLogger(root_dir)],
                         log_every_n_steps=50, # when use large batch size, need to log more often, default 50
                         limit_val_batches=0.2, # use this pct of total validation datasets
                         val_check_interval=0.5, # make validation every quarter of an epoch of training
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if get_device() in ["cuda", "mps"] else "cpu",
                         devices=1,
                         max_epochs=kwargs["max_epochs"],
                         gradient_clip_val=5)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"{kwargs['task_name']}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = TaskTrainer.load_from_checkpoint(pretrained_filename)
    else:
        model = TaskTrainer(max_iters=trainer.max_epochs*len(train_loader), **kwargs)

    trainer.fit(model, train_loader, val_loader)

    # do a final validation on all val data
    val_result = trainer.test(model, val_loader, verbose=False)
    result = {"test": val_result, "val": val_result, 
              "metrics_csv": model.trainer.log_dir if trainer.logger else None}

    model = model.to(device)
    model_path = os.path.join(CHECKPOINT_PATH, f"{kwargs['task_name']}.pth")
    model.save_checkpoint(model_path)

    return model, result

def show_attention_mat(val_loader, task_model, ax=None):
    data_input, labels = next(iter(val_loader))

    inp_data = F.one_hot(data_input, num_classes=task_model.hparams.num_classes).float()
    inp_data = inp_data.to(device)
    attention_mats = task_model.get_attention_mats(inp_data)
    plot_attention_mat(data_input, attention_mats, idx=0, ax=ax)
