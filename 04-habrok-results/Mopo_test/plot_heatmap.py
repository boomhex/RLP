import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIG ----------------
CSV_PATHS = ["good_results_big1.csv", "good_results_big2.csv", "good_results_big3.csv"]      # <- put multiple files here
TEST_COL = "loss_test"
TRAIN_COL = "loss_train"
ROLL_WIN = 2                            # 1 = no smoothing
CENTER = True                           # centered rolling mean across widths
# ---------------------------------------


def save_heatmaps_train_test_multi(
    csv_paths: list[str],
    test_col: str = TEST_COL,
    train_col: str = TRAIN_COL,
    roll_win: int = ROLL_WIN,
    center: bool = CENTER,
    out_test: str = "heatmap_test.png",
    out_train: str = "heatmap_train.png",
) -> None:
    if not csv_paths:
        raise ValueError("csv_paths is empty.")

    paths = [Path(p) for p in csv_paths]
    missing_files = [str(p) for p in paths if not p.exists()]
    if missing_files:
        raise FileNotFoundError(f"CSV not found: {missing_files}")

    # Load + concatenate into one "database" DataFrame
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source__"] = str(p)  # optional: keep provenance
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    needed = {"epoch", "width", test_col, train_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    # Ensure numeric
    for col in ["epoch", "width", test_col, train_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["epoch", "width", test_col, train_col])

    if df.empty:
        raise ValueError("No valid rows after numeric conversion / NaN drop.")

    roll_win = int(max(1, roll_win))

    def _plot_and_save(loss_col: str, out_path: str) -> None:
        # Mean across duplicates (including across files) for each (epoch,width)
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
            raise ValueError(
                f"Heatmap table is empty for {loss_col} (check your data)."
            )

        # Rolling mean along widths (use transpose to avoid deprecated axis=1)
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

        plt.colorbar(im, label=f"{loss_col} (width roll win={roll_win})")
        plt.xlabel("width")
        plt.ylabel("epoch")

        plt.xticks(
            range(len(heat.columns)),
            [str(int(w)) if float(w).is_integer() else str(w) for w in heat.columns],
            rotation=45,
            ha="right",
        )
        plt.yticks(
            range(len(heat.index)),
            [str(int(e)) if float(e).is_integer() else str(e) for e in heat.index],
        )

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    _plot_and_save(test_col, out_test)
    _plot_and_save(train_col, out_train)

    print(f"Saved: {out_test}")
    print(f"Saved: {out_train}")


# Example:
save_heatmaps_train_test_multi(CSV_PATHS, roll_win=ROLL_WIN, center=CENTER)