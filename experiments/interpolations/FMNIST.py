import torch
from torchvision import transforms
import yaml
import numpy as np
import csv
from tqdm.auto import tqdm

import sys, os

sys.path.append(os.path.abspath("../../"))

import fourierimaging as fi

from fourierimaging.utils.helpers import load_model, init_opt, fix_seed
from fourierimaging.modules import TrigonometricResize_2d, SpectralCNN, SpectralResNet
from fourierimaging.utils import datasets as data
import fourierimaging.train as train

from omegaconf import DictConfig, OmegaConf, open_dict
from select_sizing import sizing


saved_model_path = "../saved_models/"
saved_models = sorted([f for f in os.listdir(saved_model_path)])
saved_models = [
    saved_model_path + f for f in saved_models if "cnn" in f and "spectral" not in f
]

for model_path in saved_models:
    print(model_path)
    conf = torch.load(model_path)["conf"]
    """ 
    with open_dict(conf):
        conf["dataset"]["path"] = "../../../datasets"
    """
    device = conf.train.device
    # %% get train, validation and test loader
    train_loader, valid_loader, test_loader = data.load(conf["dataset"])

    model = load_model(conf).to(device)

    model.load_state_dict(torch.load(model_path)["model_state_dict"])

    model_im_shape = list(torch.load(model_path)["conf"]["im_shape"])
    # print(list(model_im_shape), type(list(model_im_shape)))
    """ # %%
    spectral = False
    if spectral:
        model = SpectralCNN.from_CNN(model, fix_out=False).to(device)
    """
    # %% eval
    data_sizing = ["TRIGO", "BILINEAR"]
    model_sizing = ["NONE", "TRIGO", "BILINEAR"]
    combinations = [(d, m) for d in data_sizing for m in model_sizing]

    fname = "results/FMNIST" + "_Michi" + ".csv"
    sizes = [13, 18, 23, 28, 33, 38, 43, 48, 53, 58]

    model.eval()
    accs = []
    for d in data_sizing:
        accs_loc = []
        print(20 * "<>")
        print("Starting test for data sizing: " + d)
        print(20 * "<>")
        for s in sizes:
            acc = 0
            tot_steps = 0
            resize_data = sizing(d, [s, s])
            # resize_model = sizing(m, model_im_shape)
            # print(s, model_im_shape)
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(test_loader):
                    # get batch data
                    x, y = x.to(device), y.to(device)

                    # resize input

                    x = resize_data(x)
                    # print(x.shape)

                    # evaluate
                    # x = resize_model(x)
                    pred = model(x)
                    acc += (pred.max(1)[1] == y).sum().item()
                    tot_steps += y.shape[0]
            print(20 * "<>")
            print("Done for s=" + str(s))
            print("Test Accuracy: " + str(100 * acc / tot_steps) + str("[%]"))
            print(20 * "<>")
            accs_loc.append(acc / tot_steps)
        accs.append([d, model_im_shape[0]] + accs_loc)

    accs = [[model_path] + col for col in accs]

    # print(accs)
    with open(fname, "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        # writer.writerow(["model name", "data sizing", "model sizing"] + list(sizes))
        for i in range(len(accs)):
            writer.writerow(accs[i])
