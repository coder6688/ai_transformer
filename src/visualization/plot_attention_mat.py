from matplotlib import pyplot as plt
import numpy as np


def plot_attention_mat(input_data, attn_maps, idx=0, ax=None):
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3

    if ax is None:
        fig, axs = plt.subplots(num_layers, num_heads, figsize=(num_heads*fig_size, num_layers*fig_size))
    else:
        if num_layers > 1 or num_heads > 1:
            raise ValueError("External axis can only be used for single-layer/single-head models")
        axs = np.array([[ax]])

    if isinstance(axs, np.ndarray):
        axs = axs.reshape(-1, num_heads)

    for layer in range(num_layers):
        for head in range(num_heads):
            # Get current axis based on input type
            if isinstance(ax, plt.Axes):  # Single axis provided
                curr_ax = ax
            elif num_layers == 1 and num_heads == 1:  # Single subplot grid
                curr_ax = axs
            else:  # Multi-subplot grid
                curr_ax = axs[layer][head] if num_heads > 1 else axs[layer]
            
            # Plot on current axis
            curr_ax.imshow(attn_maps[layer][head], origin='lower', vmin=0)
            curr_ax.set_xticks(list(range(seq_len)))
            curr_ax.set_xticklabels(input_data.tolist())
            curr_ax.set_yticks(list(range(seq_len)))
            curr_ax.set_yticklabels(input_data.tolist())
            curr_ax.set_title(f"Layer {layer+1}, Head {head+1}")

    if ax is None:  # Only adjust when we created the figure
        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        plt.suptitle(f"Attention Patterns (Sample {idx})", y=1.02)

    