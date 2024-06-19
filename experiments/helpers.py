from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from omegaconf import DictConfig, OmegaConf, open_dict
from fourierimaging.modules import TrigonometricResize_2d
import torch


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0


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
