import torch
from torch import nn
from models.ModuleRedefinitions import RelevanceNet, Layer, ReLu as PropReLu, \
    NextConvolution, FirstConvolution, BatchNorm2d, FlattenToLinearLayer, NextLinear


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
        # output = self.net(x)
        return output.view(-1, 1).squeeze(1)

    def relprop(self):
        return self.net.relprop()

    def setngpu(self, ngpu):
        self.ngpu = ngpu


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
                NextConvolution(d, 2 * d, 4, stride=2, padding=1, alpha=1),
                BatchNorm2d(2 * d),
                PropReLu(),
            ),
            Layer(
                NextConvolution(2 * d, 4 * d, 4, stride=2, padding=1, alpha=1),
                BatchNorm2d(4 * d),
                PropReLu(),
            ),
            Layer(
                NextConvolution(4 * d, 8 * d, 4, stride=2, padding=1, alpha=1),
                BatchNorm2d(8 * d),
                PropReLu(),
            ),
            Layer( # We take relevance here
                NextConvolution(8 * d, 1, 4, stride=1, padding=0, alpha=1)
            ),
            Layer(  # Output Layer
                nn.Sigmoid()
            )
        )


class CIFARDiscriminatorNet(DiscriminatorNet):
    def __init__(self, ndf, nc):
        super(CIFARDiscriminatorNet, self).__init__(ndf, nc)

        self.loss = nn.BCELoss()

        net = RelevanceNet()

        net.add_module('conv0', FirstConvolution(nc, ndf, stride=2, kernel_size=5))
        net.add_module('relu0', PropReLu())

        net.add_module('conv1', NextConvolution(ndf, ndf * 2, stride=2, kernel_size=5))
        net.add_module('bn1', BatchNorm2d(ndf * 2))
        net.add_module('relu1', PropReLu())

        net.add_module('conv2', NextConvolution(ndf * 2, ndf * 4, stride=2, kernel_size=5))
        net.add_module('bn2', BatchNorm2d(ndf * 4))
        net.add_module('relu2', PropReLu())

        net.add_module('conv3', NextConvolution(ndf * 4, ndf * 8, stride=2, kernel_size=5))
        net.add_module('bn3', BatchNorm2d(ndf * 8))
        net.add_module('relu3', PropReLu())

        net.add_module('conv4', NextConvolution(ndf * 8, ndf * 16, stride=2, kernel_size=5))
        net.add_module('bn4', BatchNorm2d(ndf * 16))
        net.add_module('relu4', PropReLu())

        net.add_module('Flatten', FlattenToLinearLayer())
        net.add_module('lastlinear', NextLinear(ndf * 16, 1))

        net.add_module('sigmoid', nn.Sigmoid())

        self.net = net



    def forward(self, input):

        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, input, range(self.ngpu))
        else:
            output = self.net(input)

        return output.view(-1, 1).squeeze(1)




class WGANDiscriminatorNet(DiscriminatorNet):

    def __init__(self, isize, nc, ndf, ngpu, n_extra_layers=0):
        super(WGANDiscriminatorNet, self).__init__(ndf, nc)
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = RelevanceNet()
        main.cuda()
        main.add_module('initial-conv{0}-{1}'.format(nc, ndf),
                       FirstConvolution(nc, ndf, 4, 2, 1))
        main.add_module('initial-relu{0}'.format(ndf),
                       PropReLu(inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                           NextConvolution(cndf, cndf, 3, 1, 1))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                           BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                           PropReLu(inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                           NextConvolution(in_feat, out_feat, 4, 2, 1, alpha=2.0))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                           BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                           PropReLu(inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # We take relevance here
        # state size. K x 4 x 4
        # Global average to single output
        main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                       NextConvolution(cndf, 1, 4, 1, 0, alpha=2.0))
        self.main = main

    def setngpu(self, ngpu):
        self.ngpu = ngpu

    def forward(self, input):

        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = output.mean(0)
        return output.view(1)

    def relprop(self):
        return self.main.relprop()

