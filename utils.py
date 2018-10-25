import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
import torch
from matplotlib import pyplot as plt
if torch.cuda.is_available():
    plt.switch_backend('agg')
import PIL, PIL.Image

'''
    TensorBoard Data will be stored in './runs' path
'''

lowest = -1.0
highest = 1.0


class Logger:

    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.comment)

    def log(self, d_error, g_error, epoch, n_batch, num_batches):

        var_class = torch.Tensor
        if type(d_error.data) == var_class:
            d_error = d_error.data.cpu().numpy()
        if type(g_error.data) == var_class:
            g_error = g_error.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/D_error'.format(self.comment), d_error, step)
        self.writer.add_scalar(
            '{}/G_error'.format(self.comment), g_error, step)

    def log_images(self, images, relevance, num_images, epoch, n_batch, num_batches, format='NCHW', normalize=True):
        """
        input images are expected in format (NCHW)
        """
        relevance = visualize(relevance.cpu().numpy() if torch.cuda.is_available()
                              else relevance.numpy(), heatmap)

        if type(images) == np.ndarray:
            images = torch.from_numpy(images)
        if type(relevance) == np.ndarray:
            relevance = torch.from_numpy(relevance)

        if format == 'NCHW':
            # images = images.transpose(1, 3)
            relevance = relevance.permute(0, 3, 1, 2)

        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.comment, '')

        # concat images
        images = images.repeat(1, 3, 1, 1)
        if torch.cuda.is_available():
            images = images.cuda()
            relevance = relevance.cuda()
        images = torch.cat((images, relevance))

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=normalize, scale_each=True, pad_value=1)
        # Make vertical grid from image tensor
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(
            images, nrow=nrows, normalize=True, scale_each=True, pad_value=1)

        # Add horizontal images to tensorboard
        self.writer.add_image(img_name, horizontal_grid, step)

        # Save plots
        self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):
        out_dir = './output/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)

        # Plot and save horizontal
        fig = plt.figure(figsize=(32, 16), facecolor='white')
        if torch.cuda.is_available():
            horizontal_grid = horizontal_grid.cpu()
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self._save_images(fig, epoch, n_batch, 'hori')
        plt.close()

        # Save squared
        # fig = plt.figure()
        # plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        # plt.axis('off')
        # self._save_images(fig, epoch, n_batch)
        # plt.close()

    def _save_images(self, fig, epoch, n_batch, comment=''):
        out_dir = './output/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir, comment, epoch, n_batch))

    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):

        var_class = torch.Tensor
        if type(d_error.data) == var_class:
            d_error = d_error.data.cpu().numpy()
        if type(g_error.data) == var_class:
            g_error = g_error.data.cpu().numpy()
        if type(d_pred_real.data) == var_class:
            d_pred_real = d_pred_real.data
        if type(d_pred_fake.data) == var_class:
            d_pred_fake = d_pred_fake.data

        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch, num_epochs, n_batch, num_batches)
        )
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f} \n'.format(d_pred_real.mean(), d_pred_fake.mean()))
    def save_models(self, generator, discriminator, epoch):
        out_dir = './data/models/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        torch.save(generator.state_dict(),
                   '{}/G_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(),
                   '{}/D_epoch_{}'.format(out_dir, epoch))

    def close(self):
        self.writer.close()

    # Private Functionality

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


# --------------------------------------
# Color maps ([-1,1] -> [0,1]^3)
# --------------------------------------

def heatmap(x):
    x = x[..., np.newaxis]

    # positive relevance
    hrp = 0.9 - np.clip(x - 0.3, 0, 0.7) / 0.7 * 0.5
    hgp = 0.9 - np.clip(x - 0.0, 0, 0.3) / 0.3 * 0.5 - np.clip(x - 0.3, 0, 0.7) / 0.7 * 0.4
    hbp = 0.9 - np.clip(x - 0.0, 0, 0.3) / 0.3 * 0.5 - np.clip(x - 0.3, 0, 0.7) / 0.7 * 0.4

    # negative relevance
    hrn = 0.9 - np.clip(-x - 0.0, 0, 0.3) / 0.3 * 0.5 - np.clip(-x - 0.3, 0, 0.7) / 0.7 * 0.4
    hgn = 0.9 - np.clip(-x - 0.0, 0, 0.3) / 0.3 * 0.5 - np.clip(-x - 0.3, 0, 0.7) / 0.7 * 0.4
    hbn = 0.9 - np.clip(-x - 0.3, 0, 0.7) / 0.7 * 0.5

    r = hrp * (x >= 0) + hrn * (x < 0)
    g = hgp * (x >= 0) + hgn * (x < 0)
    b = hbp * (x >= 0) + hbn * (x < 0)

    return np.concatenate([r, g, b], axis=-1)


def graymap(x):
    x = x[..., np.newaxis]
    return np.concatenate([x, x, x], axis=-1) * 0.5 + 0.5


# --------------------------------------
# Visualizing data
# --------------------------------------

def visualize(x, colormap):
    N = len(x)
    assert (N <= 16)
    x = colormap(x / np.abs(x).max())

    # Create a mosaic and upsample
    x = x.reshape([N, 64, 64, 3])
    return x
    # x = np.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=1)
    # x = x.transpose([0, 2, 1, 3, 4]).reshape([1 * 32, N * 32, 3])
    # x = np.kron(x, np.ones([2, 2, 1]))
    # PIL.Image.fromarray((x * 255).astype('byte'), 'RGB').save('./data/images/VGAN/MNIST/' + name)
