import matplotlib.pyplot as plt


def plot_history_lstm(history):
    """
    Plot LSTM training/validation loss curves and optionally learning rate.

    Parameters
    ----------
    history : dict
        History dict with keys:
          - 'train_loss' : list-like
          - 'val_loss'   : list-like
          - 'lr'         : list-like, optional

    Returns
    -------
    None
        Creates and shows a matplotlib figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot losses
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue")
    ax1.plot(epochs, history["val_loss"], label="Validation Loss", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # If LR is present, plot on secondary axis
    if "lr" in history:
        ax2 = ax1.twinx()
        ax2.plot(
            epochs,
            history["lr"],
            label="Learning Rate",
            color="tab:green",
            linestyle="--",
        )
        ax2.set_ylabel("Learning Rate")
        ax2.legend(loc="upper center")

    plt.show()
