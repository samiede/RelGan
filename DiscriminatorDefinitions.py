import argparse
import os
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.nn.utils import weight_norm
import utils
from utils import Logger
from ModuleRedefinitions import RelevanceNet, Layer, ReLu as PropReLu, \
    NextConvolution, FirstConvolution, Pooling, Dropout, BatchNorm2d, NextLinear


class DiscriminatorNet(nn.Module):

    def __init__(self, d, nc):
        super(DiscriminatorNet, self).__init__()

        self.net = None

    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)

    def relprop(self):
        return self.net.relprop()


class MNISTDiscriminatorNet(DiscriminatorNet):

    def __init__(self, d, nc):
        super(MNISTDiscriminatorNet, self).__init__(d, nc)

        self.net = RelevanceNet(
            Layer(  # Input Layer
                FirstConvolution(nc, d, 4, stride=2, padding=1),
                PropReLu(),
            ),
            Layer(
                NextConvolution(d, 2 * d, 4, stride=2, padding=1),
                BatchNorm2d(2 * d),
                PropReLu(),
            ),
            Layer(
                NextConvolution(2 * d, 4 * d, 4, stride=2, padding=1),
                BatchNorm2d(4 * d),
                PropReLu(),
            ),
            Layer(
                NextConvolution(4 * d, 8 * d, 4, stride=2, padding=1),
                BatchNorm2d(8 * d),
                PropReLu(),
            ),
            Layer(  # Output Layer
                NextConvolution(8 * d, 1, 4, stride=1, padding=0),
                nn.Sigmoid()
            )
        )


class CIFARDiscriminatorNet(DiscriminatorNet):

    def __init__(self, d, nc):
        super(CIFARDiscriminatorNet, self).__init__(d, nc)

        self.net = RelevanceNet(
            Layer(
                Dropout(0.2),
                weight_norm(FirstConvolution(in_channels=nc, out_channels=d, kernel_size=3, stride=1, padding=1),
                            'weight'),
                PropReLu(),
            ),
            Layer(
                weight_norm(NextConvolution(in_channels=d, out_channels=d, kernel_size=3, stride=1, padding=1),
                            'weight'),
                PropReLu(),
                Dropout(0.2),

            ),
            Layer(
                weight_norm(NextConvolution(in_channels=d, out_channels=2 * d, kernel_size=3, stride=1, padding=1),
                            'weight'),
                PropReLu(),
            ),
            Layer(
                weight_norm(NextConvolution(in_channels=2 * d, out_channels=2 * d, kernel_size=3, stride=2, padding=1),
                            'weight'),
                PropReLu(),
                Dropout(0.2),

            ),
            Layer(
                weight_norm(NextConvolution(in_channels=2 * d, out_channels=2 * d, kernel_size=3, stride=1, padding=0),
                            'weight'),
                PropReLu(),
            ),
            Layer(
                weight_norm(NextConvolution(in_channels=2 * d, out_channels= 2 * d, kernel_size=1, stride=1, padding=0),
                            'weight'),
                PropReLu(),
            ),
            Layer(
                weight_norm(NextConvolution(in_channels=2 * d, out_channels=2 * d, kernel_size=1, stride=1, padding=0),
                            'weight'),
                PropReLu(),
            ),
            Layer(
                Pooling(kernel_size=14, name='global'),
                NextLinear(in_features=2 * d, out_features=10),
            ),
            Layer(
                NextLinear(in_features=10, out_features=1),
                nn.Sigmoid()
            )
        )
