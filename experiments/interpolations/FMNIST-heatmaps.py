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

    filt = data[data["model name"].isin(["CNN25"])]
    filt = filt.drop(columns=["model name", "data sizing"])
    filt = filt.pivot(index="model im shape", columns="im shape", values="acc")
    ax = sns.heatmap(filt, annot=True)
    ax.set(xlabel="training resolution", ylabel="testing resolution")
    plt.savefig(f"CNN25-heatmap,{sizing}.pdf")
    plt.show()

    filt = data[data["model name"].isin(["CNN10"])]
    filt = filt.drop(columns=["model name", "data sizing"])
    filt = filt.pivot(index="model im shape", columns="im shape", values="acc")
    ax = sns.heatmap(filt, annot=True)
    ax.set(xlabel="training resolution", ylabel="testing resolution")
    plt.savefig(f"CNN10-heatmap,{sizing}.pdf")
    plt.show()
