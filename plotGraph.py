from pathlib import Path
import matplotlib.pyplot as plt


def plot_validation_loss_vs_width_by_height(
    results: dict,
    output_path: str | Path = "val_loss_vs_width_by_height.png",
    title: str = "Validation Loss vs Model Width",
):
    """
    Plots validation loss (Y) vs model width (X), with one line per model height.

    Parameters
    ----------
    results : dict
        {height: {width: validation_loss}}
        height = model depth / number of layers
        width  = hidden size (e.g., h_size)
    output_path : str | Path
        Where to save the figure.
    title : str
        Plot title.
    """

    output_path = Path(output_path)
    plt.figure(figsize=(8, 6))

    for height, data in sorted(results.items()):
        widths = sorted(data.keys())
        val_losses = [data[w] for w in widths]

        plt.plot(
            widths,
            val_losses,
            marker="",
            label=f"depth = {height}",
        )

    plt.xlabel("Model width")
    plt.ylabel("Validation loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()

results = {
    2: {50: 0.42, 100: 0.35, 200: 0.31},
    4: {50: 0.38, 100: 0.30, 200: 0.27},
    6: {50: 0.36, 100: 0.28, 200: 0.25},
}

plot_validation_loss_vs_width_by_height(results)