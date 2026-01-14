import pandas as pd
import matplotlib.pyplot as plt


def plot_train_test_loss_per_width_at_epochs(
    csv_paths: list[str],          # <-- multiple CSVs
    epochs: list[int],
    train_col: str = "loss_train",
    test_col: str = "loss_test",
    roll_win: int = 1,             # 1 = no smoothing
    center: bool = True,
) -> None:
    if not csv_paths:
        raise ValueError("csv_paths must contain at least 1 path")

    # ---- load + concatenate ----
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)

        needed = {"epoch", "width", train_col, test_col}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")

        for col in ["epoch", "width", train_col, test_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["epoch", "width", train_col, test_col])
        df["source_file"] = path  # optional, useful for debugging
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # ---- determine which epochs exist (in any file) ----
    epochs = [int(e) for e in epochs]
    available = set(df["epoch"].astype(int).unique())

    plt.figure()
    plotted = 0

    for ep in epochs:
        if ep not in available:
            continue

        # rows from all files at this epoch
        sub = df[df["epoch"].astype(int) == ep]

        # key point: average across files for each (epoch,width) data point
        # (and also handles duplicates within a file, if they exist)
        agg_test = sub.groupby("width")[test_col].mean().sort_index()
        agg_train = sub.groupby("width")[train_col].mean().sort_index()

        if roll_win and roll_win > 1:
            agg_test = agg_test.rolling(
                window=roll_win, center=center, min_periods=1
            ).mean()
            agg_train = agg_train.rolling(
                window=roll_win, center=center, min_periods=1
            ).mean()

        line_test, = plt.plot(
            agg_test.index.to_numpy(),
            agg_test.to_numpy(),
            marker="",
            label=f"test  epoch {ep}",
        )
        color = line_test.get_color()

        plt.plot(
            agg_train.index.to_numpy(),
            agg_train.to_numpy(),
            marker="",
            linestyle=":",
            color=color,
            label=f"train epoch {ep}",
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
    plt.title(
        f"Loss vs width (avg over {len(csv_paths)} CSVs, "
        f"rolling mean win={roll_win}, test solid / train dotted)"
    )
    plt.tight_layout()
    out_path = "line.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


# Example:
plot_train_test_loss_per_width_at_epochs(
    ["results_epochs10.csv","results_epochs11.csv","results_epochs12.csv"],
    [275,300,325,350,375,400,425],
    roll_win=,
    center=True,
)