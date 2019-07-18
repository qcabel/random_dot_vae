from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
import scipy.ndimage
import os


parser = argparse.ArgumentParser(description='VAE random dot Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epoch-size', type=int, default=100, metavar='N',
                    help='epoch size (# of batches) for generating random dots online (technically each epoch is different.')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hidden-dim', type=int, default=6, metavar='N',
                    help='dimension of the hidden space z')
parser.add_argument('--beta-latent-loss', type=float, default=1,
                    help='beta to control the weight for kl-divergence loss')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# --- DO NOT NEED MNIST ---
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

if not os.path.exists("results"):
    os.makedirs("results")


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, args.hidden_dim)
        self.fc22 = nn.Linear(400, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + args.beta_latent_loss * KLD

def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma))
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def next_random_dot_batch(batch_size, if_standard=False):
    """Random dot images generated online,
    if standard, generate one image with the dot in the middle"""

    # must be float32 array
    random_dot_imgs = np.empty([batch_size, 28*28], dtype=np.float32)
    H = matlab_style_gauss2D([15, 15], 2)  # lowpass filter

    # control the number of dots displayed
    # random integer between 1 and 10 indicating the set size: how many dots in img
    a = np.ones(batch_size, dtype=int)
    # a = np.random.randint(1, 11, self.batchsize)

    if if_standard:
        batch_size = 1
        dot_img = np.zeros([28, 28])
        dot_img[14, 14] = 1
        dot_img = scipy.ndimage.convolve(dot_img, H, mode='nearest')  # filter
        dot_img = np.reshape(dot_img, [1, 28 * 28])  # flatten
        random_dot_imgs[0,:] = dot_img / np.max(dot_img)
    else:
        for i_img in range(batch_size):
            dot_img = np.zeros([28, 28])
            coords = np.random.randint(0, 28, [a[i_img], 2])
            for x, y in coords:
                dot_img[x, y] = 1                     # one pixel dot
            dot_img = scipy.ndimage.convolve(dot_img, H, mode='nearest') # filter
            dot_img = np.reshape(dot_img, [1, 28*28]) # flatten
            dot_img = dot_img / np.max(dot_img)       # normalize to 1
            random_dot_imgs[i_img,:] = dot_img        # combine the whole batch

    return random_dot_imgs


def train(epoch):
    model.train()
    train_loss = 0

    # generate 100 mini-batch per epoch
    for batch_idx in range(args.epoch_size):
        # generate the random dot, then convert numpy array to tensor
        data = next_random_dot_batch(args.batch_size)
        # numpy -> tensor
        data = torch.from_numpy(data).to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:#interval to save training log
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), args.epoch_size * args.batch_size,
                100. * batch_idx / args.epoch_size,
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / args.epoch_size / args.batch_size))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i in range(args.epoch_size):
            data = next_random_dot_batch(args.batch_size)
            # numpy -> tensor
            data = torch.from_numpy(data).to(device)

            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                # need to view both arrays as 4d arrays (with 1-d color channel) for
                # visualization.
                comparison = torch.cat([data.view(args.batch_size, 1, 28, 28)[:n],
                                       recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png',
                           nrow=n, pad_value=1)

    test_loss /= (args.epoch_size * args.batch_size)
    print('====> Test set loss: {:.4f}'.format(test_loss))

# a standard image with one dot in the middle
standard_img = next_random_dot_batch(1, if_standard=True)
standard_img = torch.from_numpy(standard_img).to(device)

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

        with torch.no_grad():#sample from the generative model
            sample = torch.randn(64, args.hidden_dim).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png',
                       pad_value=1)

            # visualize latent space
            z_vis = torch.zeros(12*args.hidden_dim, args.hidden_dim)
            z_vis_ind = 0
            mu, logvar = model.encode(standard_img)
            z = model.reparameterize(mu, logvar)
            for idim in range(args.hidden_dim):
                z_tmp = torch.from_numpy(z.numpy().copy())
                for z_value in np.arange(-3, 3, .5):
                    z_tmp[0,idim] = z_value
                    z_vis[z_vis_ind,:] = z_tmp
                    z_vis_ind += 1
            z_vis = model.decode(z_vis).cpu()
            save_image(z_vis.view(12*args.hidden_dim, 1, 28, 28),
                       'results/zvisualize_' + str(epoch) + '.png',
                       nrow=12, pad_value=1)
