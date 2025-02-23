import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from functools import partial
from matplotlib import pyplot as plt

from config.set_seed import set_seed
from core.embedding_functions import embedding_torch
from core.get_optimizer_lr import get_optimizer_with_lr
from data.reverse_dataset import prepare_data_loader
from visualization.plot_training_history import plot_training_history
from training.training_common import show_attention_mat, training


if __name__ == "__main__":
    set_seed()

    train_loader, val_loader, test_loader = prepare_data_loader()

    num_layers = 1
    num_heads = 1
    input_dim=train_loader.dataset.num_categories
    embed_dim=train_loader.dataset.num_categories

    task_model, task_result = training(train_loader, val_loader, test_loader,
                                        task_name="ReverseTask",
                                        input_dim=input_dim,
                                        embed_dim=embed_dim,                                        
                                        model_dim=16,
                                        fea_dim=32,
                                        mlp_dim=128,
                                        num_heads=num_heads,
                                        num_classes=train_loader.dataset.num_categories,
                                        output_dim=train_loader.dataset.num_categories,
                                        num_layers=num_layers,
                                        dropout=0.1,
                                        input_dropout=0.1,
                                        max_epochs=5,
                                        lr=1e-3,
                                        warmup=50,                                                  
                                        optimizer_config=get_optimizer_with_lr("cosine_warmup"),
                                        embedding_func=partial(embedding_torch, vocab=range(train_loader.dataset.num_categories), embed_dim=embed_dim)
                                        )
    
    print(f"Val accuracy:  {(100.0 * task_result['val'][0]['test_acc']):4.2f}%")
    #print(f"Test accuracy: {(100.0 * task_result['test'][0]['test_acc']):4.2f}%")


    # Only combine plots if model has single layer/head
    if num_layers == 1 and num_heads == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        show_attention_mat(val_loader, task_model, ax=ax1)
        # when CSVLogger is enabled
        plot_training_history(task_result['metrics_csv'], ax=ax2)
        plt.tight_layout()
        fig.savefig('combined_plot.png', dpi=300)  # Save the combined figure here
        plt.show()
    else:
        # Show separate figures for multi-head/layer models
        show_attention_mat(val_loader, task_model)
        # when CSVLogger is enabled
        plot_training_history(task_result['metrics_csv'])
        plt.savefig('separate_plots.png', dpi=300)  # Save as separate figures
        plt.show()
