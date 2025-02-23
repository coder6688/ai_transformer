import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_history(training_metrics_csv, ax=None):
    assert training_metrics_csv is not None, "training_metrics_csv is None"

    print(training_metrics_csv)

    # Load CSV data
    df = pd.read_csv(training_metrics_csv + "/metrics.csv")

    # Plot setup
    if ax is None:
        ax = plt.gca()

    #plt.figure(figsize=(10, 6))
    sns.set_style('whitegrid')

    # Plot training accuracy
    ax.plot(df['step'], df['train_acc'], 
            label='Training Accuracy', 
            marker='o',
            alpha=0.7)

    # Plot test accuracy (if available)
    if 'test_acc' in df and not df['test_acc'].isnull().all():
        ax.plot(df['step'], df['test_acc'], 
                label='Test Accuracy',
                marker='s',
                linestyle='--',
                alpha=0.7)

    # Formatting
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training vs Test Accuracy', fontsize=14)
    ax.legend()
    ax.set_ylim(0, 1.05)  # Assuming accuracy is between 0-1

