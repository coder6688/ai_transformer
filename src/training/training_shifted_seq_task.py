import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch.utils.data as data
from functools import partial
from matplotlib import pyplot as plt

from config.set_seed import set_seed
from core.get_optimizer_lr import get_optimizer_with_lr
from core.embedding_functions import embedding_torch
from data.shifted_seq_dataset import ShiftedSeqDataset
from visualization.plot_training_history import plot_training_history
from training.training_common import show_attention_mat, training


def prepare_data_loader():
    dataset = partial(ShiftedSeqDataset, 10, 16)
    train_loader = data.DataLoader(dataset(50000), batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    val_loader   = data.DataLoader(dataset(1000), batch_size=128)
    test_loader  = data.DataLoader(dataset(10000), batch_size=128)

    return train_loader,val_loader, test_loader


if __name__ == "__main__":
    set_seed()
    
    train_loader, val_loader, test_loader = prepare_data_loader()

    num_layers = 1
    num_heads = 1
    embed_dim=train_loader.dataset.num_categories

    task_model, task_result = training(train_loader, val_loader, test_loader,
                                        task_name="ShiftedSeqTask",
                                        input_dim=train_loader.dataset.num_categories,                                        
                                        embed_dim=train_loader.dataset.num_categories,
                                        model_dim=32,
                                        fea_dim=32,
                                        mlp_dim=32,
                                        num_heads=1,
                                        num_classes=train_loader.dataset.num_categories,
                                        output_dim=train_loader.dataset.num_categories,
                                        num_layers=1,
                                        dropout=0.0,
                                        input_dropout=0.0,
                                        max_epochs=5,
                                        lr=5e-2,
                                        warmup=50,                                                  
                                        optimizer_config=get_optimizer_with_lr("cosine_warmup"),
                                        embedding_func=partial(embedding_torch, vocab=range(train_loader.dataset.num_categories), embed_dim=embed_dim)
                                        )
    
    print(f"Val accuracy:  {(100.0 * task_result['val'][0]['test_acc']):4.2f}%")
    #print(f"Test accuracy: {(100.0 * task_result['test'][0]['test_acc']):4.2f}%")


    # Only combine plots if model has single layer/head
    if num_layers == 1 and num_heads == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        # when tensorboard is not installed
        show_attention_mat(val_loader, task_model, ax=ax1)
        plot_training_history(task_result['metrics_csv'], ax=ax2)
        plt.tight_layout()
        fig.savefig('combined_plot.png', dpi=300)  # Save the combined figure here
        plt.show()
    else:
        # Show separate figures for multi-head/layer models
        show_attention_mat(val_loader, task_model)
        # when tensorboard is not installed
        plot_training_history(task_result['metrics_csv'])
        plt.savefig('separate_plots.png', dpi=300)  # Save as separate figures
        plt.show()
