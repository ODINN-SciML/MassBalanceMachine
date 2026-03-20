import matplotlib.pyplot as plt
import seaborn as sns
import math


def plot_training_history(history, ax=None, skip_first_n=0):
    """
    Plots loss function as function of iterations

    Parameters
    ----------
    history : attribute of NeuralNetRegressor

    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto; if None, a new figure and axes are created.

    skip_first_n: skip the first iterations in the display of the plot, optional

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if a new one was created, otherwise None.
    """
    # Skip first N entries if specified
    if skip_first_n > 0:
        history = history[skip_first_n:]

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry.get("train_loss") for entry in history]
    valid_loss = [entry.get("valid_loss") for entry in history if "valid_loss" in entry]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    else:
        fig = None

    ax.plot(epochs, train_loss, label="Training Loss")

    if valid_loss:
        # Align epochs with valid_loss length
        valid_epochs = epochs[: len(valid_loss)]
        plt.plot(valid_epochs, valid_loss, label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"Training and Validation Loss (Skipped first {skip_first_n} epochs)"
        if skip_first_n > 0
        else "Training and Validation Loss"
    )
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig
