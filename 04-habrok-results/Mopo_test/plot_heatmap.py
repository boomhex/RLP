import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ---------------- CONFIG ----------------
CSV_PATHS = ["good_results_big1.csv", "good_results_big2.csv", "good_results_big3.csv"]
TEST_COL = "loss_test"
TRAIN_COL = "loss_train"
ROLL_WIN = 4
CENTER = True

MAX_XTICKS = 12   # show only some widths
MAX_YTICKS = 12   # show only some epochs
# ---------------------------------------


def save_heatmaps_train_test_multi(
    csv_paths: list[str],
    test_col: str = TEST_COL,
    train_col: str = TRAIN_COL,
    roll_win: int = ROLL_WIN,
    center: bool = CENTER,
    out_test: str = "heatmap_test.png",
    out_train: str = "heatmap_train.png",
    max_xticks: int = MAX_XTICKS,
    max_yticks: int = MAX_YTICKS,
) -> None:
    if not csv_paths:
        raise ValueError("csv_paths is empty.")

    paths = [Path(p) for p in csv_paths]
    missing_files = [str(p) for p in paths if not p.exists()]
    if missing_files:
        raise FileNotFoundError(f"CSV not found: {missing_files}")

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source__"] = str(p)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    needed = {"epoch", "width", test_col, train_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    for col in ["epoch", "width", test_col, train_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["epoch", "width", test_col, train_col])

    if df.empty:
        raise ValueError("No valid rows after numeric conversion / NaN drop.")

    roll_win = int(max(1, roll_win))

    def _format_num(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else str(v)

    def _pick_ticks(n: int, max_ticks: int) -> np.ndarray:
        """
        Return tick positions (indices) so that at most ~max_ticks are shown.
        Always includes first and last.
        """
        if n <= 0:
            return np.array([], dtype=int)
        max_ticks = max(2, int(max_ticks))  # at least show endpoints
        if n <= max_ticks:
            return np.arange(n, dtype=int)
        step = int(np.ceil((n - 1) / (max_ticks - 1)))
        ticks = np.arange(0, n, step, dtype=int)
        if ticks[-1] != n - 1:
            ticks = np.append(ticks, n - 1)
        return ticks

    def _plot_and_save(loss_col: str, out_path: str) -> None:
        heat = (
            df.pivot_table(
                index="epoch",
                columns="width",
                values=loss_col,
                aggfunc="mean",
            )
            .sort_index()
            .sort_index(axis=1)
        )

        if heat.empty:
            raise ValueError(f"Heatmap table is empty for {loss_col} (check your data).")

        if roll_win > 1:
            heat = (
                heat.T.rolling(window=roll_win, center=center, min_periods=1)
                .mean()
                .T
            )

        plt.figure()
        im = plt.imshow(
            heat.to_numpy(),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )

        plt.colorbar(im, label=f"{loss_col}")
        plt.xlabel("width")
        plt.ylabel("epoch")
        plt.title(f"Heatmap: width vs epoch ({loss_col} HalfCheetah)")

        # ---- THIN TICKS HERE ----
        x_vals = heat.columns.to_numpy()
        y_vals = heat.index.to_numpy()

        xt = _pick_ticks(len(x_vals), max_xticks)
        yt = _pick_ticks(len(y_vals), max_yticks)

        plt.xticks(xt, [_format_num(x_vals[i]) for i in xt], rotation=45, ha="right")
        plt.yticks(yt, [_format_num(y_vals[i]) for i in yt])
        # -------------------------

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    _plot_and_save(test_col, out_test)
    _plot_and_save(train_col, out_train)

    print(f"Saved: {out_test}")
    print(f"Saved: {out_train}")


# Example:
save_heatmaps_train_test_multi(CSV_PATHS, roll_win=ROLL_WIN, center=CENTER)