import argparse
import os
import torch
from torch import nn, optim
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn
from utils import Logger

from models import GeneratorDefinitions as gd, DiscriminatorDefinitions as dd


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='MNIST | cifar10, default = MNIST', default='MNIST')
# parser.add_argument('--network', help='DCGAN | WGAN, default = DCGAN', default='DCGAN')
# parser.add_argument('--optimizer', help='adam | rmsprop, default adam', default='adam')
parser.add_argument('--imageSize', help='Size of image', type=int, default=64)
parser.add_argument('--batchSize', help='Batch size', type=int, default=64)
parser.add_argument('--ngpu', help='Number of available gpus', type=int, default=1)
parser.add_argument('--epochs', help='Number of epochs the algorithm runs', type=int, default=25)
parser.add_argument('--netf', default='./network', help='Folder to save model checkpoints')
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
cudnn.benchmark = True
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

# Maybe add more networks
def init_discriminator():
    return dd.MNISTDiscriminatorNet(ndf, nc)


def init_generator():
    return gd.MNISTGeneratorNet(ngf, nc)

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


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)

# Network Definitions

# Create Logger instance
logger = Logger(model_name='LRPGAN', data_name='MNIST')
print('Created Logger')

dataset, nc = load_dataset()

# Create Data Loader
# noinspection PyUnresolvedReferences
data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

print('Initialized Data Loader')

# number of batches
num_batches = len(data_loader)

# Create networks
discriminator = init_discriminator().to(gpu)
generator = init_generator().to(gpu)

discriminator.apply(weight_init)
generator.apply(weight_init)

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

loss = nn.BCELoss().to(gpu)

num_test_samples = 1
# We use this noise to create images during the run
test_noise = noise(num_test_samples).detach()

# Training

# Additive noise to stabilize Training for DCGAN
initial_additive_noise_var = 0.1
add_noise_var = 0.1


num_epochs = opt.epochs

for epoch in range(num_epochs):
    for n_batch, (real_batch, _) in enumerate(data_loader):
        print('Batch', n_batch, end='\r')
        n = real_batch.size(0)

        # Train Discriminator
        discriminator.zero_grad()

        y_real = discriminator_target(n).to(gpu)
        y_fake = generator_target(n).to(gpu)
        x_r = real_batch.to(gpu)

        # Predict on real data
        d_prediction_real = discriminator(x_r)
        d_loss_real = loss(d_prediction_real, y_real)

        # Create and predict on fake data
        z_ = noise(n).to(gpu)
        x_f = generator(z_).to(gpu)

        # Detach so we don't calculate the gradients here (speed up)
        d_prediction_fake = discriminator(x_f.detach())
        d_loss_fake = loss(d_prediction_fake, y_fake)
        d_training_loss = d_loss_real + d_loss_fake

        # Backpropagate and update weights
        d_training_loss.backward()
        d_optimizer.step()

        # Train Generator
        generator.zero_grad()

        # Generate and predict on fake images as if they were real
        z_ = noise(n).to(gpu)
        x_f = generator(z_)
        g_prediction_fake = discriminator(x_f)
        g_training_loss = loss(g_prediction_fake, y_real)

        # Backpropagate and update weights
        g_training_loss.backward()
        g_optimizer.step()

        # Log batch error
        logger.log(d_training_loss, g_training_loss, epoch, n_batch, num_batches)

        print('[%d/%d][%d/%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
              % (epoch, num_epochs, n_batch, num_batches, d_training_loss,
                 g_training_loss, d_loss_real, d_loss_fake))

        # Display Progress every few batches
        if n_batch % 100 == 0 or n_batch == num_batches:

            test_fake = generator(test_noise)
            test_result = discriminator(test_fake)
            test_relevance = discriminator.relprop()
            # Add up relevance of all color channels
            test_relevance = torch.sum(test_relevance, 1, keepdim=True)

            print('Test fake', test_fake.shape, 'test_rel', test_relevance.shape)
            logger.log_images(
                test_fake.data, test_relevance, num_test_samples,
                epoch, n_batch, num_batches
            )

            status = logger.display_status(epoch, num_epochs, n_batch, num_batches, d_training_loss, g_training_loss,
                                           d_prediction_real, d_prediction_fake)
            