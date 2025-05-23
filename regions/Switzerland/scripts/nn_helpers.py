import matplotlib.pyplot as plt
import os 
import seaborn as sns

def plot_training_history(custom_nn, skip_first_n=0):
    history = custom_nn.history

    # Skip first N entries if specified
    if skip_first_n > 0:
        history = history[skip_first_n:]

    epochs = [entry['epoch'] for entry in history]
    train_loss = [entry.get('train_loss') for entry in history]
    valid_loss = [entry.get('valid_loss') for entry in history if 'valid_loss' in entry]

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, train_loss, label='Training Loss', marker='o')

    if valid_loss:
        # Align epochs with valid_loss length
        valid_epochs = epochs[:len(valid_loss)]
        plt.plot(valid_epochs, valid_loss, label='Validation Loss', marker='x')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss (Skipped first {skip_first_n} epochs)" if skip_first_n > 0 else "Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # save the plot
    # Create a folder to save figures (optional)
    save_dir = "figures"
    os.makedirs(save_dir, exist_ok=True)
    # Save the figure
    plt.savefig(os.path.join(save_dir, "training_history.png"),
                dpi=300,
                bbox_inches='tight')
    plt.close()  # closes the plot to avoid display in notebooks/scripts


def predVSTruth_(grouped_ids, mae, rmse, title):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    legend_nn = "\n".join(
        (r"$\mathrm{MAE_{nn}}=%.3f, \mathrm{RMSE_{nn}}=%.3f$ " % (
            mae,
            rmse,
        ), ))

    marker_nn = 'o'
    sns.scatterplot(grouped_ids,
                    x="target",
                    y="pred",
                    ax=ax,
                    alpha=0.5,
                    marker=marker_nn)

    ax.set_ylabel('Predicted PMB [m w.e.]', fontsize=20)
    ax.set_xlabel('Observed PMB [m w.e.]', fontsize=20)

    ax.text(0.03,
            0.98,
            legend_nn,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=20)
    ax.legend([], [], frameon=False)
    # diagonal line
    pt = (0, 0)
    ax.axline(pt, slope=1, color="grey", linestyle="-", linewidth=0.2)
    ax.axvline(0, color="grey", linestyle="-", linewidth=0.2)
    ax.axhline(0, color="grey", linestyle="-", linewidth=0.2)
    ax.grid()
    ax.set_title(title, fontsize=20)
    plt.tight_layout()