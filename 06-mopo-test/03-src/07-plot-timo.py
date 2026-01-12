import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results_epochs.csv")
# epoch = 190
# depth = 4

# df = df[df["epoch"].astype(int) == epoch]
# df = df[df["depth"] == depth]



# # plt.title(f"Epoch vs Loss")
# plt.title(f"epoch={epoch}, d={depth}, Width vs Loss")
# plt.plot(df["width"], df["loss_test"], label="test")
# plt.plot(df["width"], df["loss_train"], label="train")

# plt.ylabel("MSE")
# plt.xlabel("width_param")
# plt.legend()
# plt.show()

# make sure types are numeric
df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
df["width"] = pd.to_numeric(df["width"], errors="coerce")
df["loss_test"] = pd.to_numeric(df["loss_test"], errors="coerce")
df["loss_train"] = pd.to_numeric(df["loss_train"], errors="coerce")

# optional filter (uncomment if you want a single depth)
# depth = 4
# df = df[df["depth"] == depth]

def plot_heatmap(data: pd.DataFrame, value_col: str, title: str):
    # pivot into a matrix: rows=epoch, cols=width, values=loss
    mat = data.pivot_table(index="epoch", columns="width", values=value_col, aggfunc="mean")

    plt.figure(figsize=(10, 6))
    # imshow wants a dense 2D array; NaNs are fine (they render as missing)
    im = plt.imshow(mat.values, aspect="auto", origin="lower")

    plt.colorbar(im, label=value_col)

    # label axes with actual epoch/width values
    plt.xlabel("width")
    plt.ylabel("epoch")
    plt.title(title)

    # show a readable subset of ticks
    xticks = range(len(mat.columns))
    yticks = range(len(mat.index))

    step_x = max(1, len(mat.columns) // 10)
    step_y = max(1, len(mat.index) // 10)

    plt.xticks(
        ticks=list(range(0, len(mat.columns), step_x)),
        labels=[str(int(w)) for w in mat.columns[::step_x]],
        rotation=45,
        ha="right",
    )
    plt.yticks(
        ticks=list(range(0, len(mat.index), step_y)),
        labels=[str(int(e)) for e in mat.index[::step_y]],
    )

    plt.tight_layout()
    plt.show()

# test-loss heatmap
plot_heatmap(df, "loss_test", "Heatmap: width vs epoch (test loss)")

# train-loss heatmap (optional)
plot_heatmap(df, "loss_train", "Heatmap: width vs epoch (train loss)")