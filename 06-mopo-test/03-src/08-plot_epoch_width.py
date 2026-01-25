import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATHS = [
    "results_epochs.csv"
]

WIDTH_COL = "width"
EPOCH_COL = "epoch"
TRAIN_COL = "loss_train"
TEST_COL  = "loss_test"

use_test_only = True

# choose slices:
CHOSEN_EPOCH_FOR_WIDTH_PLOT = 6000   # width plot: fix epoch
CHOSEN_WIDTH_FOR_EPOCH_PLOT = 1710   # epoch plot: fix width (nearest will be used if exact not found)

# smoothing (rolling mean over points, not over "real units")
SMOOTH_WIN_WIDTH = 8    # window in number of width points
SMOOTH_WIN_EPOCH = 40   # window in number of epoch points
SMOOTH_CENTER = True    # center the rolling window


def load_all(csv_paths):
    frames = []
    for path in csv_paths:
        path = Path(path)
        data = pd.read_csv(path)
        data["__source__"] = str(path)
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def pick_nearest_value(values: np.ndarray, target: float) -> float:
    """Return the nearest available value to `target` from `values`."""
    values = np.asarray(values, dtype=float)
    return float(values[np.argmin(np.abs(values - target))])


def smooth_out(series: pd.Series, win: int) -> pd.Series:
    s = series.rolling(window=win, center=SMOOTH_CENTER, min_periods=1).mean()
    return s


def main():
    df = load_all(CSV_PATHS)

    needed = {WIDTH_COL, EPOCH_COL, TRAIN_COL, TEST_COL}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # numeric cleanup
    for col in [WIDTH_COL, EPOCH_COL, TRAIN_COL, TEST_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[WIDTH_COL, EPOCH_COL, TRAIN_COL, TEST_COL]).copy()

    # compute error
    if use_test_only:
        df["error"] = df[TEST_COL]
        error_label = TEST_COL
    else:
        df["error"] = df[TEST_COL] - df[TRAIN_COL]
        error_label = f"{TEST_COL} - {TRAIN_COL}"

    # ---------------- Plot 1: (error + train loss) vs width at a chosen epoch ----------------
    width_slice = df[df[EPOCH_COL] == CHOSEN_EPOCH_FOR_WIDTH_PLOT].copy()
    if width_slice.empty:
        available = sorted(df[EPOCH_COL].unique())
        raise ValueError(
            f"No rows for epoch={CHOSEN_EPOCH_FOR_WIDTH_PLOT}. "
            f"Available epochs (sample): {available[:20]} ..."
        )

    width_slice = (
        width_slice.groupby(WIDTH_COL, as_index=False)
                   .agg(error=("error", "mean"), train=(TRAIN_COL, "mean"))
                   .sort_values(WIDTH_COL)
                   .reset_index(drop=True)
    )

    width_slice["error_smooth"] = smooth_out(
        width_slice["error"], SMOOTH_WIN_WIDTH
    )
    width_slice["train_smooth"] = smooth_out(
        width_slice["train"], SMOOTH_WIN_WIDTH
    )

    plt.figure()
    plt.plot(width_slice[WIDTH_COL], width_slice["error_smooth"], linewidth=1, label=error_label)
    plt.plot(width_slice[WIDTH_COL], width_slice["train_smooth"], linewidth=1, label=TRAIN_COL)

    plt.xlabel("Model width")
    plt.ylabel("Loss")
    title1 = f"Loss vs width (epoch = {CHOSEN_EPOCH_FOR_WIDTH_PLOT}, depth = 4, noise = 10%)"
    plt.title(title1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"loss_vs_width_epoch_{CHOSEN_EPOCH_FOR_WIDTH_PLOT}.png", dpi=300)
    plt.show()

    # ---------------- Plot 2: (error + train loss) vs epoch at a chosen width ----------------
    available_widths = df[WIDTH_COL].unique()
    chosen_width = float(CHOSEN_WIDTH_FOR_EPOCH_PLOT)
    nearest_width = pick_nearest_value(available_widths, chosen_width)

    if nearest_width != chosen_width:
        print(f"[info] Requested width={chosen_width}, using nearest available width={nearest_width}")

    epoch_slice = df[df[WIDTH_COL] == nearest_width].copy()
    if epoch_slice.empty:
        available = sorted(df[WIDTH_COL].unique())
        raise ValueError(
            f"No rows for width={CHOSEN_WIDTH_FOR_EPOCH_PLOT} (nearest tried: {nearest_width}). "
            f"Available widths (sample): {available[:20]} ..."
        )

    epoch_slice = (
        epoch_slice.groupby(EPOCH_COL, as_index=False)
                   .agg(error=("error", "mean"), train=(TRAIN_COL, "mean"))
                   .sort_values(EPOCH_COL)
                   .reset_index(drop=True)
    )

    epoch_slice["error_smooth"] = epoch_slice["error"].rolling(
        window=SMOOTH_WIN_EPOCH, center=SMOOTH_CENTER, min_periods=1
    ).mean()
    epoch_slice["train_smooth"] = epoch_slice["train"].rolling(
        window=SMOOTH_WIN_EPOCH, center=SMOOTH_CENTER, min_periods=1
    ).mean()

    plt.figure()
    plt.plot(epoch_slice[EPOCH_COL], epoch_slice["error_smooth"], linewidth=1, label=error_label)
    plt.plot(epoch_slice[EPOCH_COL], epoch_slice["train_smooth"], linewidth=1, label=TRAIN_COL)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    title2 = f"Loss vs epoch (width = {nearest_width:g}, depth = 4, noise = 10%)"
    plt.title(title2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"loss_vs_epoch_width_{nearest_width:g}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()