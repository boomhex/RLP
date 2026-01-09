import pandas as pd
import matplotlib.pyplot as plt

def plot_train_test_loss_per_width_at_epochs(
    csv_path: str,
    epochs: list[int],
    train_col: str = "loss_train",
    test_col: str = "loss_test",
) -> None:
    df = pd.read_csv(csv_path)

    needed = {"epoch", "width", train_col, test_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for col in ["epoch", "width", train_col, test_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["epoch", "width", train_col, test_col])

    epochs = [int(e) for e in epochs]
    available = set(df["epoch"].astype(int).unique())

    plt.figure()
    plotted = 0

    for ep in epochs:
        if ep not in available:
            continue

        sub = df[df["epoch"].astype(int) == ep]

        agg_test = (sub.groupby("width")[test_col].mean().sort_index())
        agg_train = (sub.groupby("width")[train_col].mean().sort_index())

        # Plot test first; capture the line color
        line_test, = plt.plot(
            agg_test.index.to_numpy(),
            agg_test.to_numpy(),
            marker="",
            #label=f"test  epoch {ep}",
        )
        color = line_test.get_color()

        # Plot train in same color but dotted
        plt.plot(
            agg_train.index.to_numpy(),
            agg_train.to_numpy(),
            marker="",
            linestyle=":",
            color=color,
            #label=f"train epoch {ep}",
        )

        plotted += 1

    if plotted == 0:
        avail_list = sorted(available)
        raise ValueError(
            f"None of the requested epochs {epochs} found. "
            f"Available examples: {avail_list[:20]}..."
        )

    plt.xlabel("width")
    plt.ylabel("loss")
    plt.title("Loss vs width for selected epochs (test solid, train dotted)")
    plt.legend()
    plt.tight_layout()
    out_path = "line.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

# Example:
plot_train_test_loss_per_width_at_epochs("results_epochs.csv", [375])