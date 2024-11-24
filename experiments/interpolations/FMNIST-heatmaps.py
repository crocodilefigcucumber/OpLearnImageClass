import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

palette = sns.color_palette("flare", as_cmap=True)
sns.set_style("whitegrid")
sns.set_context("paper")

data_sizing = ["TRIGO", "BILINEAR"]

for sizing in data_sizing:
    data = pd.read_csv("./results/FMNIST_Michi.csv")

    image_sizes = [
        col
        for col in data.columns.values
        if col not in ["model name", "data sizing", "model im shape"]
    ]

    y = data[data["data sizing"] == sizing]
    y = y[y["model im shape"] != 8]

    CNN10 = (
        y[data["model name"].str.contains("10-10.*202409", regex=True)]
        .drop(columns=["model name", "data sizing"])
        .sort_values("model im shape")
        .drop(columns="model im shape")
        .to_numpy()
    )

    CNN25 = (
        y[data["model name"].str.contains("25-25.*202409", regex=True)]
        .drop(columns=["model name", "data sizing"])
        .sort_values("model im shape")
        .drop(columns="model im shape")
        .to_numpy()
    )

    data = data[data["data sizing"] == sizing]
    data = data[data["model im shape"] != 8]
    data = data.sort_values("model im shape")

    data.loc[
        data["model name"].str.contains("cnn-10-10.*202409", regex=True, case=False),
        "model name",
    ] = "CNN10"
    data.loc[
        data["model name"].str.contains("cnn-25-25.*202409", regex=True, case=False),
        "model name",
    ] = "CNN25"

    data.loc[data["model name"].str.contains("CNO", case=False), "model name"] = "CNO"
    data.loc[
        data["model name"].str.contains("resnet", case=False), "model name"
    ] = "resnet"
    data.loc[data["model name"].str.contains("vit", case=False), "model name"] = "ViT"

    data = pd.melt(
        data,
        id_vars=["model name", "data sizing", "model im shape"],
        value_vars=image_sizes,
        var_name="im shape",
        value_name="acc",
    )

    data["model im shape"] = data["model im shape"].astype(int)
    data["im shape"] = data["im shape"].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    filt10 = data[data["model name"].isin(["CNN10"])]
    filt10 = filt10.drop(columns=["model name", "data sizing"])
    filt10 = filt10.pivot(index="model im shape", columns="im shape", values="acc")

    filt25 = data[data["model name"].isin(["CNN25"])]
    filt25 = filt25.drop(columns=["model name", "data sizing"])
    filt25 = filt25.pivot(index="model im shape", columns="im shape", values="acc")

    vmin, vmax = min(filt10.stack().min(), filt25.stack().min()), max(
        filt10.stack().max(), filt25.stack().max()
    )

    sns.heatmap(filt10, annot=True, ax=axes[0], vmin=vmin, vmax=vmax, cbar=False)
    axes[0].set(title="CNN10", xlabel="test resolution", ylabel="train resolution")

    sns.heatmap(filt25, annot=True, ax=axes[1], vmin=vmin, vmax=vmax, cbar=True)
    axes[1].set(title="CNN25", xlabel="test resolution", ylabel="train resolution")

    plt.tight_layout()
    plt.savefig(f"CNN-heatmap,{sizing}.pdf")
    plt.show()
