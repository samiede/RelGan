import argparse
import os
import torch
from torch import nn, optim
from torchvision import transforms, datasets
import utils
from utils import Logger
from ModuleRedefinitions import RelevanceNet, Layer, ReLu as PropReLu, \
    NextConvolution, FirstConvolution, Pooling, Dropout, BatchNorm2d

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='MNIST | cifar10', default='MNIST')
parser.add_argument('--imageSize', type=int, default=64, help='The height/width of the training/output images')
parser.add_argument('--netf', default='./Networks', help='Folder to save model checkpoints')
parser.add_argument('--netG', default='', help="Path to load generator (continue training or application)")
parser.add_argument('--netD', default='', help="Path to load discriminator (continue training or application)")
parser.add_argument('--ngd', default=64, type=int, help='Factor of generator layer depth')
parser.add_argument('--ndd', default=64, type=int, help='Factor of discriminator layer depth')

opt = parser.parse_args()
ngd = int(opt.ngd)
ndd = int(opt.ndd)
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

# CUDA everything

gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
print(gpu)


# Misc. helper functions

def load_dataset():
    if opt.dataset == 'MNIST':
        out_dir = './dataset/MNIST'
        return datasets.MNIST(root=out_dir, train=True, download=True,
                              transform=transforms.Compose(
                                  [
                                      transforms.Resize(opt.imageSize),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]
                              )), 1

    elif opt.dataset == 'cifar10':
        out_dir = './dataset/cifar10'
        return datasets.CIFAR10(root=out_dir, download=True, train=True,
                                transform=transforms.Compose([
                                    transforms.Resize(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])), 3


def noise(size):
    """

    Generates a vector of gaussian sampled random values
    """
    # noinspection PyUnresolvedReferences
    z = torch.randn((size, 100))
    # noinspection PyUnresolvedReferences
    z = torch.reshape(z, (size, 100, 1, 1))
    return z.to(gpu)


def added_gaussian(ins, is_training, stddev=0.2):
    if is_training:
        return ins + torch.Tensor(torch.randn(ins.size()).to(gpu) * stddev)
    return ins


def adjust_variance(variance, initial_variance, num_updates):
    return variance - initial_variance / num_updates


def discriminator_target(size):
    """
    Tensor containing ones, with shape = size
    """
    # noinspection PyUnresolvedReferences
    return torch.Tensor(size).uniform_(0.7, 1.2)


def generator_target(size):
    """
    Tensor containing zeroes, with shape = size
    :param size: shape of vector
    :return: zeros tensor
    """
    # noinspection PyUnresolvedReferences
    return torch.Tensor(size).uniform_(0, 0.3)


def weight_init(m):
    if type(m) == FirstConvolution or type(m) == NextConvolution or type(m) == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.zero_()
    # if type(m) == BatchNorm2d:
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.zero_()


# Network Definitions

class DiscriminatorNet(nn.Module):
    """
    Three hidden-layer discriminative neural network
    """

    def __init__(self, d=ngd):
        super(DiscriminatorNet, self).__init__()

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

        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)

    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)

    def relprop(self):
        return self.net.relprop()


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self, input_features=100, d=ngd):
        super(GeneratorNet, self).__init__()

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


# Create Logger instance
logger = Logger(model_name='LRPGAN', data_name=opt.dataset)

dataset, nc = load_dataset()

# Create Data Loader
# noinspection PyUnresolvedReferences
data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

# number of batches
num_batches = len(data_loader)

# Create networks
discriminator = DiscriminatorNet().to(gpu)
generator = GeneratorNet().to(gpu)

discriminator.apply(weight_init)
generator.apply(weight_init)

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

loss = nn.BCELoss().to(gpu)

num_test_samples = 1
# We use this noise to create images during the run
test_noise = noise(num_test_samples).detach()

# Training

# Additive noise to stabilize Training
initial_additive_noise_var = 0.1
add_noise_var = 0.1

# How many epochs do we train the model?
num_epochs = 200
for epoch in range(num_epochs):
    for n_batch, (real_batch, _) in enumerate(data_loader):
        print('Batch', n_batch, end='\r')
        n = real_batch.size(0)
        add_noise_var = adjust_variance(add_noise_var, initial_additive_noise_var, 2000)

        # ####### Train Discriminator ########
        discriminator.zero_grad()
        y_real = discriminator_target(n).to(gpu)
        y_fake = generator_target(n).to(gpu)
        x_r = real_batch.to(gpu)

        # Add noise to input
        x_rn = added_gaussian(x_r, True, add_noise_var)
        # Predict on real data
        d_prediction_real = discriminator(x_rn)
        d_loss_real = loss(d_prediction_real, y_real)

        # Create and predict on fake data
        z_ = noise(n).to(gpu)
        x_f = generator(z_).to(gpu)
        print(x_f.shape)
        exit()
        x_fn = added_gaussian(x_f, True, add_noise_var)

        # Detach so we don't calculate the gradients here (speed up)
        d_prediction_fake = discriminator(x_fn.detach())
        d_loss_fake = loss(d_prediction_fake, y_fake)
        d_training_loss = d_loss_real + d_loss_fake

        # Backpropagate and update weights
        d_training_loss.backward()
        d_optimizer.step()

        # ####### Train Generator ########
        generator.zero_grad()

        # Generate and predict on fake images as if they were real
        z_ = noise(n).to(gpu)
        x_f = generator(z_)
        x_fn = added_gaussian(x_f, True, add_noise_var)
        g_prediction_fake = discriminator(x_fn)
        g_training_loss = loss(g_prediction_fake, y_real)

        # Backpropagate and update weights
        g_training_loss.backward()
        g_optimizer.step()

        # Log batch error
        logger.log(d_training_loss, g_training_loss, epoch, n_batch, num_batches)
        # Display Progress every few batches
        if n_batch % 100 == 0 or n_batch == num_batches:
            # Create fake with fixed noise
            test_fake = generator(test_noise)
            # Classify fake data
            test_result = discriminator(test_fake)
            # Calculate SA and Relevance
            test_sensitivity = torch.autograd.grad(test_result, test_fake)[0].pow(2)
            test_relevance = discriminator.relprop()
            # Add up relevance and sensivity of all color channels
            test_relevance = torch.sum(test_relevance, 1, keepdim=True)
            test_sensitivity = torch.sum(test_sensitivity, 1, keepdim=True)

            logger.log_images(
                test_fake.data, test_relevance, num_test_samples,
                epoch, n_batch, num_batches
            )

            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_training_loss, g_training_loss, d_prediction_real, d_prediction_fake
            )

    torch.save(generator.state_dict(), '%s/generator_epoch_%d.pth' % (opt.netf, epoch))
    torch.save(discriminator.state_dict(), '%s/discriminator_epoch_%d.pth' % (opt.netf, epoch))