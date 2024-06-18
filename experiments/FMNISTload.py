from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from omegaconf import DictConfig, OmegaConf, open_dict
from fourierimaging.modules import TrigonometricResize_2d
import torch


def load_FMNIST(conf, size):
    valid = None
    if conf.name == "FashionMNIST":
        # load FMNIST
        transform = transforms.Compose(
            [transforms.ToTensor(), TrigonometricResize_2d(size)]
        )
        train = datasets.FashionMNIST(
            conf.path + "/" + str(size),
            train=True,
            download=conf.download,
            transform=transform,
        )
        test = datasets.FashionMNIST(
            conf.path + "/" + str(size),
            train=False,
            download=conf.download,
            transform=transform,
        )
    else:
        raise ValueError("Unknown dataset: " + conf.name)

    tr_loader, v_loader, te_loader = split_loader(
        train,
        test,
        valid=valid,
        batch_size=conf.batch_size,
        batch_size_test=conf.batch_size_test,
        train_split=conf.train_split,
        num_workers=conf.num_workers,
    )

    return tr_loader, v_loader, te_loader


def split_loader(
    train,
    test,
    valid=None,
    batch_size=128,
    batch_size_test=100,
    train_split=0.9,
    num_workers=1,
    seed=42,
):
    total_count = len(train)
    train_count = int(train_split * total_count)
    val_count = total_count - train_count
    generator = torch.Generator().manual_seed(seed)

    loader_kwargs = {"pin_memory": True, "num_workers": num_workers}
    if not (valid is None):
        valid_loader = DataLoader(valid, batch_size=batch_size, **loader_kwargs)
    elif val_count > 0:
        train, valid = torch.utils.data.random_split(
            train, [train_count, val_count], generator=generator
        )
        valid_loader = DataLoader(valid, batch_size=batch_size, **loader_kwargs)
    else:
        valid_loader = None
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    test_loader = DataLoader(test, batch_size=batch_size_test, **loader_kwargs)
    return train_loader, valid_loader, test_loader
