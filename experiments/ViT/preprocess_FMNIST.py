# added by Michal Mikuta on 2024-09-03
# Function to save dataset in ImageFolder format
import torch
from torchvision import datasets, transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from fourierimaging.modules import TrigonometricResize_2d
import numpy as np


def save_fashion_mnist_rgb(dataset, save_dir, classes, res=28, transform="bilin"):
    os.makedirs(save_dir, exist_ok=True)
    # Create directories for each class
    for class_name in classes:
        os.makedirs(os.path.join(save_dir, class_name), exist_ok=True)

    if transform == "bilin":
        transform = Image.BILINEAR
        # Save images in RGB format
        for idx, (img, label) in enumerate(dataset):
            img_rgb = transforms.ToPILImage()(img)
            img_rgb = img_rgb.resize((res, res), transform)
            img_rgb = img_rgb.convert("RGB")

            class_name = classes[label]
            img_save_path = os.path.join(save_dir, class_name, f"{idx}.png")
            img_rgb.save(img_save_path)

            if idx % 1000 == 0:  # Print progress every 1000 images
                print(f"Saved {idx} images to {save_dir}")
    elif transform == "trigo":
        transform = TrigonometricResize_2d((res, res))

        for idx, (img, label) in enumerate(dataset):
            img_rgb = transforms.ToPILImage()(img)
            img_rgb = np.asarray(img_rgb).reshape((28, 28))
            img_rgb = transform(img)
            img_rgb = transforms.ToPILImage()(img_rgb)
            img_rgb = img_rgb.convert("RGB")

            class_name = classes[label]
            img_save_path = os.path.join(save_dir, class_name, f"{idx}.png")
            img_rgb.save(img_save_path)

            if idx % 1000 == 0:  # Print progress every 1000 images
                print(f"Saved {idx} images to {save_dir}")

    else:
        raise ValueError(f"Wrong transform: {transform}")


resolutions = [13, 18, 23, 33, 38, 43, 48, 53, 58]

for res in resolutions:
    # Load the FashionMNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    fashion_mnist_dataset = datasets.FashionMNIST(
        root="./data", download=True, transform=transform
    )

    # Get the classes from the original dataset
    classes = fashion_mnist_dataset.classes

    # Split the dataset into training and validation sets
    train_data, val_data = train_test_split(
        fashion_mnist_dataset, test_size=0.2, random_state=42
    )

    # Save training data
    save_fashion_mnist_rgb(
        train_data,
        f"./data/FMNIST_RGB_{res}_trigo/train",
        classes,
        res=res,
        transform="trigo",
    )

    # Save validation data
    save_fashion_mnist_rgb(
        val_data,
        f"./data/FMNIST_RGB_{res}_trigo/val",
        classes,
        res=res,
        transform="trigo",
    )

    # Load the test set
    fashion_mnist_test = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Save test data
    save_fashion_mnist_rgb(
        fashion_mnist_test,
        f"./data/FMNIST_RGB_{res}_trigo/test",
        classes,
        res=res,
        transform="trigo",
    )

    print("Finished saving the dataset in ImageFolder format.")
