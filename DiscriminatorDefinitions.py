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

    def __init__(self, d, nc, ngpu=1):
        super(DiscriminatorNet, self).__init__()

        self.ngpu = ngpu
        self.net = None

    def forward(self, x):

        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)

        return output.view(-1, 1).squeeze(1)

    def relprop(self):
        return self.net.relprop()


class MNISTDiscriminatorNet(DiscriminatorNet):

    def __init__(self, d, nc):
        super(MNISTDiscriminatorNet, self).__init__(d, nc)

        self.loss = nn.BCELoss()

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


class WGANDiscriminatorNet(DiscriminatorNet):

    def __init__(self, ndf, nc, imageSize, ngpu, n_extra_layers=0):
        super(WGANDiscriminatorNet, self).__init__(ndf, nc, ngpu)

        net = RelevanceNet()
        net.add_module('initial-conv{0}-{1}'.format(nc, ndf),
                        FirstConvolution(nc, ndf, 4, 2, 1))
        net.add_module('initial-relu{0}'.format(ndf),
                        PropReLu(inplace=True))
        csize, cndf = imageSize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            net.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            NextConvolution(cndf, cndf, 3, 1, 1))
            net.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            BatchNorm2d(cndf))
            net.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            PropReLu(inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            net.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            NextConvolution(in_feat, out_feat, 4, 2, 1))
            net.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            BatchNorm2d(out_feat))
            net.add_module('pyramid-{0}-relu'.format(out_feat),
                           PropReLu(inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # We take relevance here
        # state size. K x 4 x 4
        # Global average to single output
        net.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                       NextConvolution(cndf, 1, 4, 1, 0))
        self.net = net


