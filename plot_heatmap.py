import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIG ----------------
CSV_PATH = "results_epochs.csv"
LOSS_COL = "loss_train"     # or "loss_test"
ROLL_WIN = 1                # 1 = no smoothing, 3/5/7 = smoother
CENTER = True               # centered rolling mean across widths
# ---------------------------------------

csv_path = Path(CSV_PATH)
if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

df = pd.read_csv(csv_path)

needed = {"epoch", "width", LOSS_COL}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"CSV missing columns: {sorted(missing)}")

df = df[["epoch", "width", LOSS_COL]].copy()
df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
df["width"] = pd.to_numeric(df["width"], errors="coerce")
df[LOSS_COL] = pd.to_numeric(df[LOSS_COL], errors="coerce")
df = df.dropna(subset=["epoch", "width", LOSS_COL])

if df.empty:
    raise ValueError("No valid rows after numeric conversion / NaN drop.")

ROLL_WIN = int(max(1, ROLL_WIN))

heat = (
    df.pivot_table(index="epoch", columns="width", values=LOSS_COL,
                   aggfunc="mean")
    .sort_index()
    .sort_index(axis=1)
)

if heat.empty:
    raise ValueError("Heatmap table is empty (check your data).")

# Rolling mean along widths (columns)
if ROLL_WIN > 1:
    heat = heat.rolling(window=ROLL_WIN, axis=1, center=CENTER,
                        min_periods=1).mean()

plt.figure()
im = plt.imshow(
    heat.to_numpy(),
    aspect="auto",
    origin="lower",
    interpolation="nearest",
)

plt.colorbar(im, label=f"{LOSS_COL} (width roll win={ROLL_WIN})")
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
plt.show()