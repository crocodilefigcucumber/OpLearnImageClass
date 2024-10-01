import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
from neuralop.models import FNO2d

import torch.nn as nn
import torch.nn.functional as F


# FNO2d Classifier
class FNO2dClassifier(FNO2d):
    """2-Dimensional Fourier Neural Operator Classifier

    n_modes_width : int
        number of modes to keep in Fourier Layer, along the width
    n_modes_height : int
        number of Fourier modes to keep along the height
    n_classes: int
        number of classes for the linear classification layer
    size: int
        image size
    """

    def __init__(
        self,
        n_modes_height,
        n_modes_width,
        hidden_channels,
        in_channels=1,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        output_scaling_factor=None,
        max_n_modes=None,
        fno_block_precision="full",
        non_linearity=F.gelu,
        stabilizer=None,
        use_channel_mlp=False,
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        n_classes=10,
        size=28,
        **kwargs,
    ):
        super().__init__(
            n_modes_height=n_modes_height,
            n_modes_width=n_modes_width,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=1,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            output_scaling_factor=output_scaling_factor,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            joint_factorization=joint_factorization,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
        )

        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.size = size

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(size * size, n_classes))

    def forward(self, x):
        x = super().forward(x, output_shape=(self.size, self.size))
        x = self.classifier(x)
        return x

    def name(self):
        return f"FNO-width-{self.n_modes_width}-height-{self.n_modes_height}-hidden-{self.hidden_channels}-n_layers-{self.n_layers}"
