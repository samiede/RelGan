import torch
from torch import nn
import utils
import copy


class FirstConvolution(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2, dilation=1, groups=1,
                 bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # Variables for Relevance Propagation
        self.X = None

    def forward(self, input):
        # Input shape: minibatch x in_channels, iH x iW
        self.X = input
        return super().forward(input)

    def relprop(self, R):

        if type(R) is tuple:

            R, params = R

            gamma, var, eps, beta, mean = params['gamma'], params['var'], params['eps'], params['beta'], \
                                          params['mean']
            var = torch.div(torch.ones(1), (torch.sqrt(var + eps)))

            iself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            iself.load_state_dict(self.state_dict())
            # Include positive biases as neurons
            iself_biases = copy.deepcopy(iself.bias.data)
            iself_biases = beta + gamma * (iself_biases - mean) * gamma
            iself.bias.data *= 0
            iself.weight.data = iself.weight.data * gamma.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(iself.weight) \
                                * var.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(iself.weight)

            nself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            nself.load_state_dict(self.state_dict())
            # Include positive biases as neurons
            nself_biases = copy.deepcopy(nself.bias.data)
            nself_biases = beta + gamma * (nself_biases - mean) * gamma
            nself.bias.data *= 0
            nself.weight.data = nself.weight.data * gamma.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(nself.weight) \
                                * var.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(nself.weight)
            nself.weight.data = torch.min(torch.Tensor(1).zero_(), nself.weight)

            pself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            pself.load_state_dict(self.state_dict())
            # Include positive biases as neurons to normalize over
            pself_biases = copy.deepcopy(pself.bias.data)
            # incorporate batch norm
            pself_biases = beta + gamma * (pself_biases - mean) * gamma
            pself.bias.data *= 0
            pself.weight.data = pself.weight.data * gamma.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(pself.weight) \
                                * var.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(pself.weight)
            pself.weight.data = torch.max(torch.Tensor(1).zero_(), pself.weight)

            X = self.X
            L = self.X * 0 + utils.lowest
            H = self.X * 0 + utils.highest

            iself_f = iself.forward(X)
            # Expand bias for addition
            iself_biases = torch.max(torch.Tensor(1).zero_(), iself_biases).unsqueeze(0).unsqueeze(2).unsqueeze(
                3).expand_as(iself_f)
            iself_f = iself_f + iself_biases
            pself_f = pself.forward(L)
            # Expand bias for addition
            pself_biases = torch.max(torch.Tensor(1).zero_(), pself_biases).unsqueeze(0).unsqueeze(2).unsqueeze(
                3).expand_as(pself_f)
            pself_f = pself_f + pself_biases
            nself_f = nself.forward(H)
            # Expand bias for addition
            nself_biases = torch.max(torch.Tensor(1).zero_(), nself_biases).unsqueeze(0).unsqueeze(2).unsqueeze(
                3).expand_as(nself_f)
            nself_f = nself_f + nself_biases

            Z = iself_f - pself_f - nself_f + 1e-9
            S = R / Z

            iself_b = torch.autograd.grad(iself_f, X, S, retain_graph=True)[0]
            pself_b = torch.autograd.grad(pself_f, L, S, retain_graph=True)[0]
            nself_b = torch.autograd.grad(nself_f, H, S)[0]

            R = X * iself_b - L * pself_b - H * nself_b

        else:

            iself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            iself.load_state_dict(self.state_dict())
            # Include positive biases as neurons
            iself_biases = copy.deepcopy(iself.bias.data)
            iself.bias.data *= 0

            nself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            nself.load_state_dict(self.state_dict())
            # Include positive biases as neurons
            nself_biases = copy.deepcopy(nself.bias.data)
            nself.bias.data *= 0
            nself.weight.data = torch.min(torch.Tensor(1).zero_(), nself.weight)

            pself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            pself.load_state_dict(self.state_dict())
            # Include positive biases as neurons
            pself_biases = copy.deepcopy(pself.bias.data)
            pself.bias.data *= 0
            pself.weight.data = torch.max(torch.Tensor(1).zero_(), pself.weight)

            X = self.X
            L = self.X * 0 + utils.lowest
            H = self.X * 0 + utils.highest

            iself_f = iself.forward(X)
            # Expand bias for addition
            iself_biases = torch.max(torch.Tensor(1).zero_(), iself_biases).unsqueeze(0).unsqueeze(2).unsqueeze(
                3).expand_as(iself_f)
            iself_f = iself_f + iself_biases
            pself_f = pself.forward(L)
            # Expand bias for addition
            pself_biases = torch.max(torch.Tensor(1).zero_(), pself_biases).unsqueeze(0).unsqueeze(2).unsqueeze(
                3).expand_as(pself_f)
            pself_f = pself_f + pself_biases
            nself_f = nself.forward(H)
            # Expand bias for addition
            nself_biases = torch.max(torch.Tensor(1).zero_(), nself_biases).unsqueeze(0).unsqueeze(2).unsqueeze(
                3).expand_as(nself_f)
            nself_f = nself_f + nself_biases

            Z = iself_f - pself_f - nself_f + 1e-9
            S = R / Z

            iself_b = torch.autograd.grad(iself_f, X, S, retain_graph=True)[0]
            pself_b = torch.autograd.grad(pself_f, L, S, retain_graph=True)[0]
            nself_b = torch.autograd.grad(nself_f, H, S)[0]

            R = X * iself_b - L * pself_b - H * nself_b

        return R.detach()


class NextConvolution(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2, dilation=1, groups=1,
                 bias=True, alpha=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # Variables for Relevance Propagation
        self.X = None
        self.alpha = alpha
        self.beta = alpha - 1

    def forward(self, input):
        # Input shape: minibatch x in_channels, iH x iW
        self.X = input
        return super().forward(input)

    def relprop(self, R):

        # Is the layer before Batch Norm?
        if type(R) is tuple:
            R, params = R

            gamma, var, eps, beta, mean = params['gamma'], params['var'], params['eps'], params['beta'], \
                                          params['mean']
            var = torch.div(torch.ones(1), (torch.sqrt(var + eps)))

            # Positive weights
            pself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            pself.load_state_dict(self.state_dict())

            # Include positive biases as neurons to normalize over
            pself_biases = copy.deepcopy(pself.bias.data)
            pself_biases = beta + gamma * (pself_biases - mean) * gamma
            pself.bias.data *= 0
            pself.weight.data = pself.weight.data * gamma.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(pself.weight) \
                                * var.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(pself.weight)
            pself.weight.data = torch.max(torch.Tensor([1e-9]), pself.weight)

            # Negative weights
            nself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            nself.load_state_dict(self.state_dict())

            # Include positive biases as neurons to normalize over
            nself_biases = copy.deepcopy(pself.bias.data)
            nself_biases = beta + gamma * (nself_biases - mean) * gamma
            nself.bias.data *= 0
            nself.weight.data = nself.weight.data * gamma.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(nself.weight) \
                                * var.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(nself.weight)
            nself.weight.data = torch.min(torch.Tensor([-1e-9]), nself.weight)

            X = self.X + 1e-9

            ZA = pself(X)
            # expand biases for addition
            pself_biases = torch.max(torch.Tensor(1).zero_(), pself_biases).unsqueeze(0).unsqueeze(2).unsqueeze(
                3).expand_as(ZA)
            ZA = ZA + pself_biases
            SA = self.alpha * torch.div(R, ZA)

            ZB = nself(X)
            # expand biases for addition HERE NEGATIVE BIASES? torch.min???
            nself_biases = torch.min(torch.Tensor(1).zero_(), nself_biases).unsqueeze(0).unsqueeze(2).unsqueeze(
                3).expand_as(ZB)
            ZB = ZB + nself_biases
            SB = - self.beta * torch.div(R, ZB)

            C = torch.autograd.grad(ZA, self.X, SA)[0] + torch.autograd.grad(ZB, self.X, SB)[0]
            R = self.X * C

        # If not, continue as usual
        else:

            pself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            pself.load_state_dict(self.state_dict())
            # Include positive biases as neurons
            pself_biases = copy.deepcopy(pself.bias.data)
            pself.bias.data *= 0
            pself.weight.data = torch.max(torch.Tensor([1e-9]), pself.weight)

            nself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            nself.load_state_dict(self.state_dict())
            # Include positive biases as neurons
            nself_biases = copy.deepcopy(pself.bias.data)
            nself.bias.data *= 0
            nself.weight.data = torch.min(torch.Tensor([-1e-9]), nself.weight)

            X = self.X + 1e-9

            ZA = pself(X)
            # expand biases for addition
            pself_biases = torch.max(torch.Tensor(1).zero_(), pself_biases).unsqueeze(0).unsqueeze(2).unsqueeze(
                3).expand_as(ZA)
            ZA = ZA + pself_biases
            SA = self.alpha * torch.div(R, ZA)

            ZB = nself(X)
            # expand biases for addition HERE NEGATIVE BIASES? torch.min???
            nself_biases = torch.min(torch.Tensor(1).zero_(), nself_biases).unsqueeze(0).unsqueeze(2).unsqueeze(
                3).expand_as(ZB)
            ZB = ZB + nself_biases
            SB = - self.beta * torch.div(R, ZB)

            C = torch.autograd.grad(ZA, self.X, SA)[0] + torch.autograd.grad(ZB, self.X, SB)[0]
            R = self.X * C

        return R.detach()


class ReLu(nn.ReLU):

    # def __init__(self, inplace=False):
    #     super().__init__(inplace=inplace)
    #
    # def forward(self, input):
    #     output = super().forward(input)
    #     # if output.sum().item() == 0:
    #     #     print('Relu input', input),
    #     #     print('Output', output)
    #     return super().forward(input)

    def relprop(self, R):
        return R


class BatchNorm2d(nn.BatchNorm2d):

    def relprop(self, R):
        return R, self.getParams()

    def getParams(self):
        return {'gamma': copy.deepcopy(self.weight), 'var': copy.deepcopy(self.running_var),
                'eps': copy.deepcopy(self.eps), 'beta': copy.deepcopy(self.bias),
                'mean': copy.deepcopy(self.running_mean)}


class Dropout(nn.Dropout):

    def __init__(self, p=0.5, inplace=False):
        super().__init__(p, inplace)

    #
    # def forward(self, input):
    #     output = super().forward(input)
    #     return super().forward(input)

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

        return input

    def relprop(self):
        R = self.relevanceOutput
        # For all layers except the last
        for layer in self[-2::-1]:
            R = layer.relprop(R)
        return R


class Layer(nn.Sequential):
    # def __init__(self, *args):
    #     super().__init__(*args)
    #
    # def forward(self, input):
    #     return super().forward(input)

    def relprop(self, R):
        for layer in self[::-1]:
            R = layer.relprop(R)
        return R


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

        for k, hook in self._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm) and hook.name == 'weight':
                remove_weight_norm(self)

        V = torch.max(torch.Tensor(1).zero_(), self.weight)
        Z = torch.matmul(self.X, torch.t(V)) + 1e-9
        S = R / Z
        C = torch.matmul(S, V)
        R = self.X * C

        return R


class LastLinear(NextLinear):

    def forward(self, input):
        input = torch.reshape(input, (input.size(0), 1, input.size(2) * input.size(3)))
        self.X = input
        return super().forward(input)


class FlattenLayer(nn.Module):

    def forward(self, input):
        return input.view(-1, 1)


class FlattenToLinearLayer(nn.Module):

    def forward(self, input):
        return input.squeeze(-1).squeeze(-1)


class Pooling(nn.AvgPool2d):

    def __init__(self, kernel_size, name=''):
        super().__init__(kernel_size)
        self.name = name
        self.X = None

    def forward(self, input):
        self.X = input
        output = super().forward(input) * self.kernel_size
        if self.name == 'global':
            output = output.squeeze()
        return output

    def relprop(self, R):
        if self.name == 'global':
            R.unsqueeze(-1).unsqueeze(-1)

        Z = (self.forward(self.X) + 1e-9)
        S = R / Z
        C = torch.autograd.grad(Z, self.X, S)[0]
        R = self.X * C
        return R


class ReshapeLayer(nn.Module):

    def __init__(self, filters, height, width):
        super().__init__()
        self.filters = filters
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(-1, self.filters, self.height, self.width)
