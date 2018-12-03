import argparse
import os
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from utils import Logger
import GeneratorDefinitions as gd
import DiscriminatorDefinitions as dd
from ModuleRedefinitions import RelevanceNet, Layer, ReLu as PropReLu, \
    NextConvolution, FirstConvolution, Pooling, Dropout, BatchNorm2d

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='MNIST | cifar10, default = MNIST', default='MNIST')
parser.add_argument('--network', help='DCGAN | WGAN, default = DCGAN', default='DCGAN')
parser.add_argument('--optimizer', help='adam | rmsprop, default adam', default='adam')
parser.add_argument('--imageSize', help='Size of image', type=int, default=64)
parser.add_argument('--Diters', type=int, default=1, help='number of D iters per each G iter, default = 1')
parser.add_argument('--netf', default='./Networks', help='Folder to save model checkpoints')
parser.add_argument('--netG', default='', help="Path to load generator (continue training or application)")
parser.add_argument('--netD', default='', help="Path to load discriminator (continue training or application)")
parser.add_argument('--ngf', default=64, type=int, help='Factor of generator filters')
parser.add_argument('--ndf', default=64, type=int, help='Factor of discriminator filters')

opt = parser.parse_args()
ngf = int(opt.ngf)
ndf = int(opt.ndf)
print(opt)

try:
    os.makedirs(opt.netf)
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

    elif opt.dataset == 'lsun':
        out_dir = './dataset/lsun'
        return datasets.LSUN(root=out_dir, classes=['bedroom_train'],
                             transform=transforms.Compose([
                                 transforms.Resize(opt.imageSize),
                                 transforms.CenterCrop(opt.imageSize),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])), 3

    elif opt.dataset == 'cifar10':
        out_dir = './dataset/cifar10'
        return datasets.CIFAR10(root=out_dir, download=True, train=True,
                                transform=transforms.Compose([
                                    transforms.Resize(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])), 3

    raise ValueError('No valid dataset found in {}'.format(opt.dataset))


def init_discriminator():
    if opt.network == 'DCGAN':
        return dd.MNISTDiscriminatorNet(ndf, nc)

    elif opt.network == 'WGAN':
        return dd.WGANDiscriminatorNet(ndf, nc, opt.imageSize)

    raise ValueError('No valid dataset found in {}'.format(opt.dataset))


def init_generator():
    if opt.network == 'DCGAN':
        return gd.MNISTGeneratorNet(ngf, nc)

    elif opt.network == 'WGAN':
        return gd.WGANGeneratorNet(ngf, nc, opt.imageSize)

    raise ValueError('No valid dataset found in {}'.format(opt.dataset))


def init_optimizer(network):
    if opt.optimizer == 'adam':
        return optim.Adam(network.parameters(), lr=0.0002, betas=(0.5, 0.999))
    else:
        return optim.RMSprop(network.parameters(), lr=0.00005)


def init_loss():
    if opt.network == 'WGAN':
        return lambda input, target: input.mean(0).view(1)
    else:
        return nn.BCELoss()


def noise(size):
    """

    Generates a vector of gaussian sampled random values
    :type size: object
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
    return torch.Tensor(size).uniform_(0.9, 1.1)


def generator_target(size):
    """
    Tensor containing zeroes, with shape = size
    :param size: shape of vector
    :return: zeros tensor
    """
    # noinspection PyUnresolvedReferences
    return torch.Tensor(size).zero_()
    # return torch.Tensor(size).uniform_(0, 0.3)


def fake_grad():
    if opt.network == 'WGAN':
        return torch.Tensor([1]) * -1
    else:
        return None


def real_grad():
    if opt.network == 'WGAN':
        return torch.Tensor([1])
    else:
        return None


def weight_init(m):
    if type(m) == FirstConvolution or type(m) == NextConvolution or type(m) == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    # if type(m) == BatchNorm2d:
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.zero_()


# Create Logger instance
logger = Logger(model_name='LRPGAN', data_name=opt.dataset)

dataset, nc = load_dataset()

# Create Data Loader
# noinspection PyUnresolvedReferences
data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

# number of batches
num_batches = len(data_loader)

# Create networks
discriminator = init_discriminator().to(gpu)
generator = init_generator().to(gpu)

discriminator.apply(weight_init)
generator.apply(weight_init)

d_optimizer = init_optimizer(discriminator)
g_optimizer = init_optimizer(generator)

loss = init_loss()

num_test_samples = 1
# We use this fixed noise to create images during the run
test_noise = noise(num_test_samples).detach()

# Training

# Additive noise to stabilize Training
initial_additive_noise_var = 0.1
add_noise_var = 0.1

# Variable definitions
n = None
y_real = None
d_training_loss = None
d_prediction_real = None
d_prediction_fake = None
d_loss_real = None
d_loss_fake = None

# How many epochs do we train the model?
num_epochs = 25
gen_iterations = 0
for epoch in range(num_epochs):
    data_iter = iter(data_loader)
    n_batch = 0
    while n_batch < num_batches:

        add_noise_var = adjust_variance(add_noise_var, initial_additive_noise_var, 2000)

        if (gen_iterations < 25 or gen_iterations % 500 == 0) and opt.Diters != 1:
            Diters = 100
        else:
            Diters = opt.Diters
        d = 0
        while d < Diters and n_batch < len(data_loader):

            # ####### Train Discriminator ########
            # (1)
            # ####### Train Discriminator ########

            # train the discriminator Diters times
            d += 1
            n_batch += 1
            data = data_iter.next()

            # train with real
            x_r, _ = data
            x_r = x_r.to(gpu)
            n = x_r.size(0)

            discriminator.zero_grad()
            y_real = discriminator_target(n).to(gpu)
            y_fake = generator_target(n).to(gpu)

            # Add noise to input
            x_rn = added_gaussian(x_r, True, add_noise_var)

            # Predict on real data
            d_prediction_real = discriminator(x_rn)
            d_loss_real = loss(d_prediction_real, y_real)
            d_loss_real.backward(real_grad())

            # Create and predict on fake data
            z_ = noise(n).to(gpu)
            x_f = generator(z_).to(gpu)
            x_fn = added_gaussian(x_f, True, add_noise_var)

            # Detach so we don't calculate the gradients here (speed up)
            d_prediction_fake = discriminator(x_fn.detach())
            d_loss_fake = loss(d_prediction_fake, y_fake)
            d_loss_fake.backward(fake_grad())

            if opt.network == 'WGAN':
                d_training_loss = d_loss_real - d_loss_fake
            else:
                d_training_loss = d_loss_real + d_loss_fake

            # Backpropagate and update weights
            # d_training_loss.backward()
            d_optimizer.step()

            if opt.network == 'WGAN':
                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

        # ####### Train Generator ########
        # (2)
        # ####### Train Generator ########
        generator.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        z_ = noise(n).to(gpu)
        x_f = generator(z_)
        x_fn = added_gaussian(x_f, True, add_noise_var)
        g_prediction_fake = discriminator(x_fn)
        g_training_loss = loss(g_prediction_fake, y_real)

        g_training_loss.backward(real_grad())
        g_optimizer.step()

        gen_iterations += 1

        # Log batch error
        logger.log(d_training_loss, g_training_loss, epoch, n_batch, num_batches)

        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
              % (epoch, num_epochs, n_batch, num_batches, gen_iterations,
                 d_training_loss, g_training_loss, d_loss_real, d_loss_fake))

        # Display Progress every few batches
        if n_batch % 100 == 0 or n_batch == num_batches:
            # Create fake with fixed noise
            generator.eval()
            test_fake = generator(test_noise)
            generator.train()
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

    if epoch % 5 == 0:
        torch.save(generator.state_dict(), '%s/generator_epoch_%d.pth' % (opt.netf, epoch))
        torch.save(discriminator.state_dict(), '%s/discriminator_epoch_%d.pth' % (opt.netf, epoch))
