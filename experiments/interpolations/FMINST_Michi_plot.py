import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

palette = sns.color_palette("flare",as_cmap=True)
sns.set_style("whitegrid")
sns.set_context("paper")


data_sizing = ["TRIGO","BILINEAR"]

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
        y[data["model name"].str.contains("10-10")]
        .drop(columns=["model name", "data sizing"])
        .sort_values("model im shape")
        .drop(columns="model im shape")
        .to_numpy()
    )
    CNN10 = [CNN10[i, i] for i in range(CNN10.shape[0])]

    CNN25 = (
        y[data["model name"].str.contains("25-25")]
        .drop(columns=["model name", "data sizing"])
        .sort_values("model im shape")
        .drop(columns="model im shape")
        .to_numpy()
    )
    CNN25 = [CNN25[i, i] for i in range(CNN25.shape[0])]

    CNO = (
        y[y["model name"].str.contains("CNO")]
        .drop(columns=["model name", "model im shape", "data sizing"])
        .to_numpy()[0, :]
    )
    resnet = (
        y[y["model name"].str.contains("resnet")]
        .drop(columns=["model name", "model im shape", "data sizing"])
        .to_numpy()[0, :]
    )


    plt.figure()
    plt.title(f"CNO/resnet VS CNN on native resolution,{sizing}")
    plt.plot(image_sizes, CNN10, label="CNN10")
    plt.plot(image_sizes, CNN25, label="CNN25")
    plt.plot(image_sizes, CNO, label="CNO")
    plt.plot(image_sizes, resnet, label="resnet")
    plt.legend()
    plt.savefig(f"CNOresnet VS CNN on native resolution,{sizing}.pdf")
    plt.show()


    data = data[data["data sizing"] == sizing]
    data = data[data["model im shape"] != 8]
    data = data.sort_values("model im shape")

    data.loc[data["model name"].str.contains("cnn-10", case=False), "model name"] = "CNN10"
    data.loc[data["model name"].str.contains("cnn-25", case=False), "model name"] = "CNN25"

    data.loc[data["model name"].str.contains("CNO", case=False), "model name"] = "CNO"
    data.loc[data["model name"].str.contains("resnet", case=False), "model name"] = "resnet"

    data = pd.melt(
        data,
        id_vars=["model name", "data sizing", "model im shape"],
        value_vars=image_sizes,
        var_name="im shape",
        value_name="acc",
    )

    data["model im shape"] = data["model im shape"].astype(int)
    data["im shape"] = data["im shape"].astype(int)


    filt = data[data["model name"] != "CNN25"]
    style_order = ["CNO", "resnet", "CNN10"]
    size_order = style_order

    sns.relplot(
        data=filt,
        kind="line",
        x="im shape",
        y="acc",
        hue="model im shape",
        style="model name",
        style_order=style_order,
        # size="model name",
        # size_order=size_order,
        palette=palette,
        legend="full",
    )
    plt.title(f"CNO/resnet vs CNN10,{sizing}")
    plt.savefig(f"CNOresnet vs CNN10,{sizing}.pdf")
    plt.show()

    filt = data[data["model name"] != "CNN10"]
    style_order = ["CNO", "resnet", "CNN25"]
    size_order = style_order

    sns.relplot(
        data=filt,
        kind="line",
        x="im shape",
        y="acc",
        hue="model im shape",
        style="model name",
        style_order=style_order,
        # size="model name",
        # size_order=size_order,
        palette=palette,
        legend="full",
    )
    plt.title(f"CNO/resnet vs CNN25,{sizing}")
    plt.savefig(f"CNOresnet vs CNN25,{sizing}.pdf")
    plt.show()