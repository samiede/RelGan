import torch
from torch import nn, optim
from torchvision import transforms, datasets
import utils
from utils import Logger
from ModuleRedefinitions import RelevanceNet, Layer, FirstLinear, NextLinear, LastLinear, ReLu as PropReLu, \
    NextConvolution, FirstConvolution, Pooling, Dropout, BatchNorm2d

# CUDA everything

gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)


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

    Generates a 1-d vector of gaussian sampled random values
    """
    # noinspection PyUnresolvedReferences
    z = torch.randn((size, 100))
    # noinspection PyUnresolvedReferences
    z = torch.reshape(z, (size, 100, 1, 1))
    return z


def discriminator_target(size):
    """
    Tensor containing ones, with shape = size
    """
    # noinspection PyUnresolvedReferences
    return torch.ones(size, 1)


def generator_target(size):
    """
    Tensor containing zeroes, with shape = size
    :param size: shape of vector
    :return: zeros tensor
    """
    # noinspection PyUnresolvedReferences
    return torch.zeros(size, 1)


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
                BatchNorm2d(d),
            ),
            Layer(
                NextConvolution(d, 2 * d, 4, stride=2, padding=1),
                PropReLu(),
                BatchNorm2d(2 * d),
            ),
            Layer(
                NextConvolution(2 * d, 4 * d, 4, stride=2, padding=1),
                PropReLu(),
                BatchNorm2d(4 * d),
            ),
            Layer(
                NextConvolution(4 * d, 1, 4, stride=1, padding=0),
                PropReLu(),
            ),
            Layer(  # Output Layer
                LastLinear(25, n_out),
                nn.Sigmoid()
            )
        )


        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)

    def forward(self, x):
        return self.net(x)

    def relprop(self, R):
        return self.net.relprop(R)

    def weight_init(self, mean, std):
        for m in self.net.modules():
            if isinstance(m, FirstConvolution) or isinstance(m, NextConvolution):
                m.weight.data.normal_(mean, std)
                # m.bias.data.fill_(0)


    def training_iteration(self, real_data, fake_data, optimizer):
        N = real_data.size(0)

        # Reset gradients
        optimizer.zero_grad()

        # 1.1 Train on real data
        prediction_real = self.forward(real_data)
        error_real = loss(prediction_real, discriminator_target(N)).detach()
        # error_real.backward()

        # 1.2 Train on fake data
        predictions_fake = self.forward(fake_data)
        error_fake = loss(predictions_fake, generator_target(N)).detach()
        # error_fake.backward()

        training_loss = loss(prediction_real - predictions_fake, discriminator_target(N))
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

        input_features = 100

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
                nn.ReLU()
            ),
            Layer(
                #               C_in, c_out,k, s, p
                nn.ConvTranspose2d(d, 1, 4, 2, 1),
                nn.Tanh()
            )
        )

    def forward(self, x):
        return self.main(x)

    def weight_init(self, mean, std):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.fill_(0)



    @staticmethod
    def training_iteration(data_fake, data_real, optimizer):
        n = data_fake.size(0)

        # Reset gradients
        optimizer.zero_grad()

        # Reshape for prediction
        data_fake_d = torch.reshape(data_fake, (100, 1, 64, 64))
        # forward pass on discriminator with generated data
        prediction_fake = discriminator(data_fake)

        # forward pass on discriminator with real data
        prediction_real = discriminator(data_real)

        # Calculate error to supposed real labels and backprop
        prediction_error = loss(prediction_fake - prediction_real, discriminator_target(n))
        prediction_error.backward()

        # Update weights with gradient
        optimizer.step()

        return prediction_error


# Create Logger instance
logger = Logger(model_name='LRPGAN', data_name='MNIST')

data = load_mnist_data()

# Create Data Loader
# noinspection PyUnresolvedReferences
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# number of batches
num_batches = len(data_loader)

# Create networks
discriminator = DiscriminatorNet().to(gpu)
generator = GeneratorNet().to(gpu)
discriminator.weight_init(0, 0.2)
generator.weight_init(0, 0.02)

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCEWithLogitsLoss().to(gpu)

num_test_samples = 1
# We use this noise to create images during the run
test_noise = noise(num_test_samples).detach()

# Training

# How often does the discriminator train on the data before the generator is trained again
d_steps = 1

num_epochs = 200

for epoch in range(num_epochs):
    for n_batch, (real_batch, _) in enumerate(data_loader):
        print('Batch', n_batch)
        n = real_batch.size(0)


        # Images for Discriminator

        # Create fake data and detach the Generator, so we don't compute the gradients here
        z = noise(n)
        fake_data = generator(z).detach()
        fake_data, real_batch = fake_data.to(gpu), real_batch.to(gpu)

        # Train Discriminator
        d_error, d_pred_real, d_pred_fake = discriminator.training_iteration(real_batch, fake_data, d_optimizer)

        fake_data = generator(noise(n))
        fake_data = fake_data.to(gpu)

        # Train Generator
        g_error = generator.training_iteration(fake_data, real_batch, g_optimizer)

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
