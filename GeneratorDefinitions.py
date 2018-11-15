import argparse
import os
import torch
from torch import nn, optim
from torch.nn.utils import weight_norm
import utils
from utils import Logger
from ModuleRedefinitions import RelevanceNet, Layer, ReLu as PropReLu, \
    NextConvolution, FirstConvolution, Pooling, Dropout, BatchNorm2d, ReshapeLayer, FlattenToLinearLayer


class MNISTGeneratorNet(torch.nn.Module):

    def __init__(self, d, nc, input_features=100):
        super(MNISTGeneratorNet, self).__init__()

        self.main = nn.Sequential(
            Layer(
                #                   Channel_in,     c_out, k, s, p
                nn.ConvTranspose2d(input_features, d * 8, 4, 1, 0),
                nn.BatchNorm2d(d * 8),
                nn.LeakyReLU(0.2)
                # state size = 100 x 1024 x 4 x 4
            ),
            Layer(
                #                   C_in, c_out,k, s, p
                nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1),
                nn.BatchNorm2d(d * 4),
                nn.LeakyReLU(0.2)
                # state size = 100 x 512 x 8 x 8
            ),
            Layer(
                #                C_in, c_out,k, s, p
                nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1),
                nn.BatchNorm2d(d * 2),
                nn.LeakyReLU(0.2)
                # state size = 100 x 256 x 16 x 16
            ),
            Layer(
                #                C_in, c_out,k, s, p
                nn.ConvTranspose2d(d * 2, d, 4, 2, 1),
                nn.BatchNorm2d(d),
                nn.LeakyReLU(0.2)

            ),
            Layer(
                #               C_in, c_out,k, s, p
                nn.ConvTranspose2d(d, nc, 4, 2, 1),
                nn.Tanh()
            )
        )

    def forward(self, x):
        return self.main(x)


class CIFARGeneratorNet(torch.nn.Module):

    def __init__(self, d, nc, input_features=100):
        super(CIFARGeneratorNet, self).__init__()

        self.main = nn.Sequential(
            Layer(
                FlattenToLinearLayer(),
                nn.Linear(in_features=input_features, out_features=512 * 4 * 4),
                nn.BatchNorm1d(512 * 4 * 4),
                nn.LeakyReLU(0.2),
            ),
            ReshapeLayer(filters=512, height=4, width=4),
            Layer(
                nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(256),
            ),
            Layer(
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(128),
            ),
            Layer(
                weight_norm(nn.ConvTranspose2d(128, nc, 5, 2, 1), 'weight'),
                nn.Tanh()
            )
        )

    def forward(self, x):
        return self.main(x)
