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
    CNN10 = [CNN10[i, i] for i in range(CNN10.shape[0])]

    CNN25 = (
        y[data["model name"].str.contains("25-25.*202409", regex=True)]
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
    ViT = (
        y[y["model name"].str.contains("vit")]
        .drop(columns=["model name", "model im shape", "data sizing"])
        .to_numpy()[0, :]
    )
    FNO = (
        y[y["model name"].str.contains("FNO")]
        .drop(columns=["model name", "model im shape", "data sizing"])
        .to_numpy()[0, :]
    )

    filt = pd.DataFrame(index=image_sizes)
    filt["CNN10"] = CNN10
    filt["CNN25"] = CNN25
    filt["ResNet"] = resnet
    filt["ViT"] = ViT
    filt["CNO"] = CNO
    filt["FNO"] = FNO
    
    plt.figure()
    sns.heatmap(data=filt.transpose(),square=True,cbar=True,annot=True)
    plt.xlabel("Test Resolution")
    plt.savefig(f"ModelsVsCNNsOnNativeResolutions,{sizing}")
    plt.show()
    # plt.title(f"CNO/resnet/ViT/FNO VS CNN on native resolution,{sizing}")
    
    plt.plot(image_sizes, CNN10, label="CNN10")
    plt.plot(image_sizes, CNN25, label="CNN25")
    plt.plot(image_sizes, CNO, label="CNO")
    plt.plot(image_sizes, resnet, label="ResNet")
    plt.plot(image_sizes, ViT, label="ViT")
    plt.plot(image_sizes, FNO, label="FNO")

    plt.legend()
    plt.xlabel("Test Resolution")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(f"CNOresnetViTFNO VS CNN on native resolution,{sizing}.pdf")
    plt.show()
    
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
    ] = "ResNet"
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
    
    filt = data[data["model name"] != "CNN25"]
    style_order = ["CNO", "resnet", "ViT", "CNN10"]
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
    #    plt.title(f"CNO/resnet/ViT vs CNN10,{sizing}")
    plt.xlabel("Test Resolution")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    #plt.savefig(f"CNOresnetViT vs CNN10,{sizing}.pdf")
    plt.show()

    filt = data[data["model name"] != "CNN10"]
    style_order = ["CNO", "resnet", "ViT", "CNN25"]
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
    # plt.title(f"CNO/resnet/ViT vs CNN25,{sizing}")
    plt.xlabel("Test Resolution")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    #plt.savefig(f"CNOresnetViT vs CNN25,{sizing}.pdf")
    plt.show()

    filt = data[data["model name"] != "CNN10"]
    fitl = filt[filt["model name"] != "CNN25"]
    style_order = ["CNO", "resnet", "ViT"]
    size_order = style_order
    sns.relplot(
        data=filt,
        kind="line",
        x="im shape",
        y="acc",
        # hue="model im shape",
        style="model name",
        style_order=style_order,
        # size="model name",
        # size_order=size_order,
        palette=palette,
        legend="full",
    )
    # plt.title(f"CNO/resnet/ViT,{sizing}")
    plt.xlabel("Test Resolution")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    #plt.savefig(f"CNOresnetViT,{sizing}.pdf")
    plt.show()

    filt = data[data["model name"].isin(["CNN10"])]
    style_order = ["CNN10"]
    size_order = style_order
    sns.relplot(
        data=filt,
        kind="line",
        x="im shape",
        y="acc",
        hue="model im shape",
        # style="model name",
        style_order=style_order,
        # size="model name",
        # size_order=size_order,
        palette=palette,
        legend="full",
    )
    # plt.title(f"CNN10,{sizing}")
    plt.xlabel("Test Resolution")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    #plt.savefig(f"CNN10,{sizing}.pdf")

    filt = data[data["model name"].isin(["CNN25"])]
    style_order = ["CNN25"]
    size_order = style_order
    sns.relplot(
        data=filt,
        kind="line",
        x="im shape",
        y="acc",
        hue="model im shape",
        # style="model name",
        style_order=style_order,
        # size="model name",
        # size_order=size_order,
        palette=palette,
        legend="full",
    )
    # plt.title(f"CNN25,{sizing}")
    plt.xlabel("Test Resolution")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    #plt.savefig(f"CNN25,{sizing}.pdf")
    plt.show()
    '''    
    plt.figure()
    data.rename(columns={"model name": "Model", "im shape": "Test Resolution", "acc": "Accuracy"}, inplace = True)
    filt = data[["Model","Test Resolution","Accuracy"]]
    print(filt.Model)
    filt = filt.pivot(index="Model",columns="Test Resolution",values="Accuracy")
    sns.heatmap(data=filt,annot=True, square = True, cbar = True)
    plt.tight_layout()
    plt.savefig(f"Model-heatmap,{sizing}.pdf")
    plt.show()
    '''
