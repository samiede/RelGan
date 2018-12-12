import torch
from torch import nn
from models.ModuleRedefinitions import Layer


class MNISTGeneratorNet(torch.nn.Module):

    def __init__(self, ngf, nc, input_features=100):
        super(MNISTGeneratorNet, self).__init__()

        self.main = nn.Sequential(
            Layer(
                #                   Channel_in,     c_out, k, s, p
                nn.ConvTranspose2d(input_features, ngf * 8, 4, 1, 0),
                nn.BatchNorm2d(ngf * 8),
                nn.LeakyReLU(0.2)
                # state size = 100 x 1024 x 4 x 4
            ),
            Layer(
                #                   C_in, c_out,k, s, p
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
                nn.BatchNorm2d(ngf * 4),
                nn.LeakyReLU(0.2)
                # state size = 100 x 512 x 8 x 8
            ),
            Layer(
                #                C_in, c_out,k, s, p
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
                nn.BatchNorm2d(ngf * 2),
                nn.LeakyReLU(0.2)
                # state size = 100 x 256 x 16 x 16
            ),
            Layer(
                #                C_in, c_out,k, s, p
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2)

            ),
            Layer(
                #               C_in, c_out,k, s, p
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
                nn.Tanh()
            )
        )

    def forward(self, x):
        return self.main(x)


class WGANGeneratorNet(torch.nn.Module):

    def __init__(self, ngf, nc, imageSize, input_features=100, n_extra_layers=0, ngpu=1):
        super(WGANGeneratorNet, self).__init__()
        self.ngpu = ngpu
        assert imageSize % 16 == 0, "imageSize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != imageSize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()

        main.add_module('initial-{0}-{1}-convt'.format(input_features, cngf),
                        nn.ConvTranspose2d(input_features, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < imageSize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,  range(self.ngpu))
        else:
            output = self.main(input)
        return output
