import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_validation_sweep_like_example(
    csv_paths,
    width_col: str = "width",
    epoch_col: str = "epoch",
    val_col: str = "loss_val",
    roll_win: int = 1,
    center: bool = True,
    log_epoch_color: bool = True,
    best_line: str = "min",
    save_path: str | None = None,
) -> None:
    # Accept single path or list of paths
    if isinstance(csv_paths, (str, bytes, Path)):
        csv_paths = [csv_paths]

    paths = [Path(p) for p in csv_paths]
    missing_files = [str(p) for p in paths if not p.exists()]
    if missing_files:
        raise FileNotFoundError(f"CSV not found: {missing_files}")

    # Load + concatenate -> one "database" df
    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        d["__source__"] = str(p)  # optional: provenance
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)

    needed = {width_col, epoch_col, val_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for c in [width_col, epoch_col, val_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[width_col, epoch_col, val_col])
    if df.empty:
        raise ValueError("No valid rows after numeric conversion / NaN drop.")

    # Mean across duplicates (including across files) for each (epoch,width)
    heat = (
        df.pivot_table(
            index=epoch_col,
            columns=width_col,
            values=val_col,
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    if heat.empty:
        raise ValueError("Pivot table is empty; check width/epoch/value columns.")

    # Keep only certain epochs (1, step, 2*step, ...)
    step = 50
    wanted = {1} | set(range(step, int(heat.index.max()) + 1, step))
    heat = heat.loc[heat.index.astype(int).isin(wanted)]

    # Rolling mean over widths (avoid deprecated axis=1)
    roll_win = int(max(1, roll_win))
    if roll_win > 1:
        heat = (
            heat.T.rolling(window=roll_win, center=center, min_periods=1)
            .mean()
            .T
        )

    epochs = heat.index.to_numpy()
    widths = heat.columns.to_numpy()

    fig, ax = plt.subplots()

    # Color mapping by epoch
    if log_epoch_color:
        ep_for_color = epochs.astype(float).copy()
        if np.any(ep_for_color <= 0):
            ep_for_color = ep_for_color - ep_for_color.min() + 1.0
        color_values = np.log10(ep_for_color)
        cbar_label = "Epochs (log)"
    else:
        color_values = epochs.astype(float)
        cbar_label = "Epochs"

    norm = plt.Normalize(color_values.min(), color_values.max())
    cmap = plt.get_cmap("viridis")

    # Plot one curve per epoch (faint), colored by epoch
    for i, _ep in enumerate(epochs):
        y = heat.iloc[i].to_numpy()
        ax.plot(
            widths,
            y,
            linewidth=1,
            alpha=0.25,
            color=cmap(norm(color_values[i])),
        )

    # "Best" line across epochs at each width
    mat = heat.to_numpy()
    if best_line == "min":
        best = np.nanmin(mat, axis=0)
        best_label = "Best (min across epochs)"
    elif best_line.startswith("q"):
        q = float(best_line[1:])
        best = np.nanpercentile(mat, q, axis=0)
        best_label = f"Best (q{q} across epochs)"
    else:
        raise ValueError("best_line must be 'min' or like 'q10', 'q25', ...")

    ax.plot(
        widths,
        best,
        linestyle="--",
        linewidth=2,
        color="red",
        label=best_label,
    )

    # Proper colorbar attached to this axes
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=cbar_label)

    ax.set_xlabel("Width")
    ax.set_ylabel("Validation Error")
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# Example:
plot_validation_sweep_like_example(
    ["good_results_big1.csv", "good_results_big2.csv", "good_results_big3.csv"],
    val_col="loss_test",
    roll_win=4,
    save_path="line_heat.png",
)