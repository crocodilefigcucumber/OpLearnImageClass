import torch
import yaml
import time

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra

# custom imports
# %% custom imports
from inspect import getsourcefile
import os.path as path, sys

current_dir = path.dirname(path.abspath(getsourcefile(lambda: 0)))
sys.path.insert(0, current_dir[: current_dir.rfind(path.sep)])

import fourierimaging as fi

from fourierimaging.utils.helpers import load_model, init_opt, fix_seed
from fourierimaging.utils import datasets as data
import fourierimaging.train as train
from helpers import load_FMNIST, EarlyStopping


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig) -> None:
    # %% fix random seed
    fix_seed(conf.seed)

    # %% get train, validation and test loader
    train_loader, valid_loader, test_loader = load_FMNIST(
        conf.dataset, conf["im_shape"]
    )

    # %% define the model
    if conf.CUDA.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda" + ":" + str(conf.CUDA.cuda_device))
        print("GPU WORKING")
    else:
        device = "cpu"

    with open_dict(conf):
        conf.train["device"] = str(device)

    model = load_model(conf).to(device)

    # %% Initialize optimizer and lamda scheduler
    opt, lr_scheduler = init_opt(model, conf["train"]["opt"])
    # initalize history
    tracked = ["train_loss", "train_acc", "val_loss", "val_acc"]
    history = {key: [] for key in tracked}
    trainer = train.trainer(
        model, opt, lr_scheduler, train_loader, valid_loader, conf["train"]
    )
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # print(OmegaConf.to_yaml(conf))
    # %%
    print(50 * "#")
    print("Starting training.")
    print("Model name " + model.name())
    print("Total number of params: " + str(total_params) + " parameters")
    print("Number of trainable params: " + str(total_trainable_params))

    early_stopping = EarlyStopping(patience=conf.train.patience, verbose=True)
    til_epoch = 0

    for i in range(conf["train"]["epochs"]):
        print(50 * ".")
        print("Starting epoch: " + str(i))
        print("Learning rate: " + str(opt.param_groups[0]["lr"]))

        # train_step
        train_data = trainer.train_step()

        # validation step
        val_data = trainer.validation_step()

        # update history
        for key in tracked:
            if key in val_data:
                history[key].append(val_data[key])
            if key in train_data:
                history[key].append(train_data[key])

        early_stopping(val_data["val_loss"], model)

        if early_stopping.early_stop:
            print("Early stopping")
            til_epoch = i
            break

    tester = train.Tester(test_loader, conf.train)
    tester(model)

    if conf.train.save:
        time_str = time.strftime("%Y%m%d-%H%M%S")
        save_name = conf.train.save_dir
        save_name += (
            model.name()
            + "imshape-"
            + str(conf["im_shape"])
            + "-"
            + "epoch-"
            + str(til_epoch)
            + "-"
            + time_str
        )
        torch.save(
            {
                "conf": conf,
                "history": history,
                "model_state_dict": model.state_dict(),
                "im_shape": conf["im_shape"],
                "trained_til_epoch": til_epoch,
                "early_stopping_patience": conf.train.patience,
            },
            save_name,
        )


if __name__ == "__main__":
    main()
