import torch
from torch import nn
import utils


class FirstLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)

        # Variables for Relevance Propagation
        self.X = None

    def forward(self, input):
        self.X = input
        return super().forward(input)

    def relprop(self, R):
        W = self.weight
        V = torch.max(torch.DoubleTensor(1).zero_(), self.weight)
        U = torch.min(torch.DoubleTensor(1).zero_(), self.weight)
        X = self.X
        L = self.X * 0 + utils.lowest
        H = self.X * 0 + utils.highest

        Z = torch.matmul(X, torch.t(W)) - torch.matmul(L, torch.t(V)) - torch.matmul(H, torch.t(U)) + 1e-9
        S = R / Z
        R = X * torch.matmul(S, W) - L * torch.matmul(S, V) - H * torch.matmul(S, U)
        return R.detach().numpy()


class NextLinear(nn.Linear):

    # Disable Bias
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)

        # Variables for Relevance Propagation
        self.X = None

    def forward(self, input):
        self.X = input
        return super().forward(input)

    def relprop(self, R):
        V = torch.max(torch.Tensor(1).zero_(), self.weight)
        Z = torch.matmul(self.X, torch.t(V)) + 1e-9
        S = R / Z
        C = torch.matmul(S, V)
        R = self.X * C
        return R


class LastLinear(NextLinear):

    def forward(self, input):
        print(input.shape)
        input = torch.reshape(input, (input.size(0), 1, input.size(2) * input.size(3)))
        self.X = input
        return super().forward(input)


class FlattenLayer(nn.Module):

    def forward(self, input):
        return input.view(-1, 1)


class FirstConvolution(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2, dilation=1, groups=1,
                 bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # Variables for Relevance Propagation
        self.X = None

    def forward(self, input):
        # Input shape: minibatch x in_channels, iH x iW
        self.X = input
        output = super().forward(input)
        # print('First Convolution Output Zero: ', output.sum().item() == 0)
        return output

    def relprop(self, R):
        iself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        iself.load_state_dict(self.state_dict())
        iself.bias.data *= 0

        nself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        nself.load_state_dict(self.state_dict())
        nself.bias.data *= 0
        nself.weight.data = torch.min(torch.Tensor(1).zero_(), nself.weight)

        pself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        pself.load_state_dict(self.state_dict())
        pself.bias.data *= 0
        pself.weight.data = torch.max(torch.Tensor(1).zero_(), pself.weight)

        X = self.X
        L = self.X * 0 + utils.lowest
        H = self.X * 0 + utils.highest

        iself_f = iself.forward(X)
        pself_f = pself.forward(L)
        nself_f = nself.forward(H)

        Z = iself_f - pself_f - nself_f + 1e-9
        S = R / Z

        iself_b = torch.autograd.grad(iself_f, X, S, retain_graph=True)[0]
        pself_b = torch.autograd.grad(pself_f, L, S, retain_graph=True)[0]
        nself_b = torch.autograd.grad(nself_f, H, S)[0]

        R = X * iself_b - L * pself_b - H * nself_b
        return R.detach()


class NextConvolution(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2, dilation=1, groups=1,
                 bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # Variables for Relevance Propagation
        self.X = None

    def forward(self, input):
        # Input shape: minibatch x in_channels, iH x iW
        self.X = input
        output = super().forward(input)
        # print('Next Convolution Output Zero', output.sum().item() == 0)
        return output

    def relprop(self, R):
        # print('Next Convolution', 'Incoming Relevance Zero: ', R.sum().item() == 0)
        pself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        pself.load_state_dict(self.state_dict())
        pself.bias.data *= 0
        pself.weight.data = torch.max(torch.Tensor(1).zero_(), pself.weight)
        Z = pself(self.X) + 1e-9
        S = torch.div(R, Z)
        C = torch.autograd.grad(Z, self.X, S)[0]
        R = self.X * C
        # print('Next Convolution Outgoing Relevance Zero: ', R.sum().item() == 0)
        return R


class ReLu(nn.ReLU):

    def __init__(self, inplace=False):
        super().__init__(inplace=inplace)

    def forward(self, input):
        output = super().forward(input)
        # print('Relu output zero: ', output.sum().item() == 0)
        if (output.sum().item() == 0):
            print('Relu input', input),
            print('Output', output)
        return super().forward(input)

    def relprop(self, R):
        return R


class Pooling(nn.AvgPool2d):

    def __init__(self, kernel_size):
        super().__init__(kernel_size)
        self.X = None

    def forward(self, input):
        self.X = input
        return super().forward(input) * self.kernel_size

    def relprop(self, R):
        Z = (self.forward(self.X) + 1e-9)
        S = R / Z
        C = torch.autograd.grad(Z, self.X, S)[0]
        R = self.X * C
        return R


class BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features):
        super().__init__(num_features)
        self.factor = None

    def forward(self, input):
        X = input
        output = super().forward(input)
        self.factor = torch.div(output, X)
        self.factor.detach()
        recovered_x = torch.div(output, self.factor)
        # self.recover(input)
        # print('recovered: ', X.sum().item() == recovered_x.sum().item(), self)
        # print(self.factor)
        return output

    def relprop(self, R):
        # self.recover(R)
        # return torch.div(R, self.factor)
        return R

    def recover(self, input):
        print('input: ', input.shape)
        print('bias:', self.bias.shape)
        print('weight', self.weight.shape)
        self.expand(self.weight, input.size)
        print('var', self.running_var.shape)
        exit()
        # denom = input - bias
        # shift = torch.div(input - self.bias, self.weight)
        # factor = torch.sqrt(self.running_var + self.eps)
        # addendum = self.running_mean
        # return shift * factor + addendum

    def expand(self, input, shape):
        expanded = torch.Tensor(shape(2), shape(3))
        expanded.fill_(input[0].item())
        torch.unsqueeze(expanded, 1)
        print('Exp',expanded.shape)
        for scalar in input[1:]:
            fill = torch.Tensor(shape(2), shape(3))
            fill.fill_(scalar.item())
            print(fill.shape)
            print('Fill', fill)
            torch.unsqueeze(fill, 0)
            expanded = torch.stack((expanded, fill), 0)
            print('Expanded', expanded.shape)




class Dropout(nn.Dropout):

    def __init__(self, p=0.5, inplace=False):
        super().__init__(p, inplace)

    def forward(self, input):
        output = super().forward(input)
        return super().forward(input)

    def relprop(self, R):
        return R


class RelevanceNet(nn.Sequential):

    def __init__(self, *args):
        super().__init__(*args)
        self.relevanceOutput = None

    def forward(self, input):

        self.relevanceOutput = None

        for idx, layer in enumerate(self):
            input = layer.forward(input)

            # save output of second-to-last layer to use in relevance propagation
            if idx == len(self) - 2:
                self.relevanceOutput = input
                if input.size()[0] == 1:
                    print('Relevance Output', self.relevanceOutput)

        return input

    def relprop(self, R):
        # print(R)
        # For all layers except the last
        for layer in self[-2::-1]:
            R = layer.relprop(R)
        return R


class Layer(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input):
        return super().forward(input)

    def relprop(self, R):
        for layer in self[::-1]:
            R = layer.relprop(R)
        return R
