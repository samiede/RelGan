import torch
from torch import nn, optim
from torchvision import transforms, datasets
import utils
from utils import Logger
from ModuleRedefinitions import RelevanceNet, Layer, FirstLinear, NextLinear, FlattenLayer, ReLu as PropReLu, \
    NextConvolution, FirstConvolution, Pooling, Dropout, BatchNorm2d

# CUDA everything

gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

print(gpu)

# Misc. helper functions

def load_mnist_data():
    transform = transforms.Compose(
        [transforms.Resize(64),
         transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
         ]
    )
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=transform, download=True)


def images_to_vectors(images):
    return images.view(images.size(0), 784)


def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


def noise(size):
    """

    Generates a vector of gaussian sampled random values
    """
    # noinspection PyUnresolvedReferences
    z = torch.randn((size, 100))
    # noinspection PyUnresolvedReferences
    z = torch.reshape(z, (size, 100, 1, 1))
    return z.to(gpu)


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
    if type(m) == FirstConvolution or type(m) == NextConvolution or type(m) == BatchNorm2d or type(m) == nn.ConvTranspose2d:
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


# Network Definitions

class DiscriminatorNet(nn.Module):
    """
    Three hidden-layer discriminative neural network
    """

    def __init__(self, d=128):
        super(DiscriminatorNet, self).__init__()

        n_out = 1
        self.net = RelevanceNet(
            Layer(  # Input Layer
                FirstConvolution(1, d, 4, stride=2, padding=1),
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
        return self.net(x).view(-1,1).squeeze(1)

    def relprop(self, R):
        return self.net.relprop(R)

    def training_iteration(self, real_data, fake_data, optimizer):
        N = real_data.size(0)

        # Reset gradients
        optimizer.zero_grad()

        # 1.1 Train on real data
        prediction_real = self.forward(real_data)
        # Calculate error & backpropagation
        error_real = loss(prediction_real, discriminator_target(N))
        # error_real.backward()
        # 1.2 Train on fake data
        predictions_fake = self.forward(fake_data)
        # Calculate error & backprop
        error_fake = loss(predictions_fake, generator_target(N))
        # error_fake.backward()
        training_loss = error_real + error_fake
        training_loss.backward()

        # 1.3 update weights
        optimizer.step()

        return error_fake + error_real, prediction_real, predictions_fake


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self, input_features=100, d=128):
        super(GeneratorNet, self).__init__()

        self.main = nn.Sequential(
            Layer(
                #                   Channel_in,     c_out, k, s, p
                nn.ConvTranspose2d(input_features, d * 8, 4, 1, 0),
                nn.BatchNorm2d(d*8),
                nn.ReLU()
                # state size = 100 x 1024 x 4 x 4
            ),
            Layer(
                #                   C_in, c_out,k, s, p
                nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1),
                nn.BatchNorm2d(d * 4),
                nn.ReLU()
                # state size = 100 x 512 x 8 x 8
            ),
            Layer(
                #                C_in, c_out,k, s, p
                nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1),
                nn.BatchNorm2d(d * 2),
                nn.ReLU()
                # state size = 100 x 256 x 16 x 16
            ),
            Layer(
                #                C_in, c_out,k, s, p
                nn.ConvTranspose2d(d * 2, d, 4, 2, 1),
                nn.BatchNorm2d(d),
                nn.ReLU(0.2)
                ),
            Layer(
                #               C_in, c_out,k, s, p
                nn.ConvTranspose2d(d, 1, 4, 2, 1),
                nn.Tanh()
            )
        )

    def forward(self, x):
        return self.main(x)


    @staticmethod
    def training_iteration(data_fake, optimizer):
        n = data_fake.size(0)

        # Reset gradients
        optimizer.zero_grad()

        # Reshape for prediction
        # forward pass on discriminator with generated data
        prediction = discriminator(data_fake)

        # Calculate error to supposed real labels and backprop
        prediction_error = loss(prediction, discriminator_target(n))
        prediction_error.backward()

        # Update weights with gradient
        optimizer.step()

        return prediction_error

# Create Logger instance
logger = Logger(model_name='LRPGAN', data_name='MNIST')

data = load_mnist_data()

# Create Data Loader
# noinspection PyUnresolvedReferences
data_loader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)
# number of batches
num_batches = len(data_loader)

# Create networks
discriminator = DiscriminatorNet().to(gpu)
generator = GeneratorNet().to(gpu)

discriminator.apply(weight_init)
generator.apply(weight_init)


d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5,0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5,0.999))

loss = nn.BCELoss().to(gpu)

num_test_samples = 1
# We use this noise to create images during the run
test_noise = noise(num_test_samples)

# Training

# How often does the discriminator train on the data before the generator is trained again
d_steps = 1

num_epochs = 200

for epoch in range(num_epochs):
    for n_batch, (real_batch, _) in enumerate(data_loader):
        print('Batch', n_batch, end='\r')
        n = real_batch.size(0)

        # Images for Discriminator

        # Create fake data and detach the Generator, so we don't compute the gradients here
        z = noise(n)
        fake_data = generator(z)
        fake_data, real_batch = fake_data.to(gpu), real_batch.to(gpu)

        # Train Discriminator
        d_error, d_pred_real, d_pred_fake = discriminator.training_iteration(real_batch, fake_data, d_optimizer)
        # Train Generator
        exit()
        fake_data = generator(noise(n))
        fake_data = fake_data.to(gpu)

        g_error = generator.training_iteration(fake_data, g_optimizer)

        # Log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        # Display Progress every few batches
        if n_batch % 100 == 0:
            test_fake = generator(test_noise)
            discriminator.eval()
            test_result = discriminator(test_fake)
            discriminator.train()
            test_relevance = discriminator.relprop(discriminator.net.relevanceOutput)

            logger.log_images(
                test_fake.data, test_relevance, num_test_samples,
                epoch, n_batch, num_batches
            )

            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )
