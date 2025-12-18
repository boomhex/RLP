import ast
import pathlib
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

DATA_DIR = pathlib.Path("./data")
FILES = ["safefile.txt", "safefile-2.txt", "safefile-3.txt"]
WINDOW = 2  # rolling mean window size


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def rolling_mean(values, window: int):
    """
    Rolling mean with a fixed window size.
    For window=2: y_smooth[i] = mean(values[i-1], values[i]) for i>=1
    We keep the output the same length by leaving the first value unsmoothed.
    """
    if window <= 1 or len(values) == 0:
        return values

    out = [values[0]]
    for i in range(1, len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i+1]) / (i - start + 1))
    return out


# -------------------------------------------------------------------
# Load and merge data
# -------------------------------------------------------------------

data = {}  # outer_key -> {x: y}

for fname in FILES:
    path = DATA_DIR / fname
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    parsed = ast.literal_eval(text)  # safe for Python literals

    for outer_key, inner_dict in parsed.items():
        data.setdefault(outer_key, {})
        data[outer_key].update(inner_dict)


# -------------------------------------------------------------------
# Plot (raw + rolling mean)
# -------------------------------------------------------------------

plt.figure(figsize=(10, 5))

for outer_key in sorted(data.keys()):
    inner = data[outer_key]
    xs = sorted(inner.keys())
    ys = [inner[x] for x in xs]

    ys_rm = rolling_mean(ys, WINDOW)

    # smoothed line
    plt.plot(xs, ys_rm, marker="o", linewidth=1, markersize=3,
             label=f"key={outer_key} (rolling mean {WINDOW})")

plt.xlabel("Inner key")
plt.ylabel("Value")
plt.title(f"Values with rolling mean window={WINDOW}")
plt.legend()
plt.tight_layout()
plt.show()