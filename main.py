import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt
import os
import scipy.ndimage
from scipy.misc import imsave as ims
from utils import *
from ops import *

class LatentAttention():
    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.n_samples = self.mnist.train.num_examples  #number of training images in MNIST (55000)
        self.n_hidden = 500                             #?
        self.n_z = 20                                   #number of latent dimensions?
        self.batchsize = 100                            #batchsize within epoch

        self.images = tf.placeholder(tf.float32, [None, 784])# 28*28 = 784
        image_matrix = tf.reshape(self.images,[-1, 28, 28, 1])
        z_mean, z_stddev = self.recognition(image_matrix)
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 28*28])

        # 1) the reconstruction loss: negative log prob of the input under the reconstructed Bernoulli distribution
        #    induced by the decoder in the data space
        #    "this can be interpreted as the number of nats required for reconstructing the input when the activation
        #    in latent space is given"
        #    1e-8 to avoid evaluation of log(0.0)
        #    ?? where to take expectation??
        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat)
                                              + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)

        # 2) the latent loss, defined as kl-divergence between the distribution in latent space induced by the encoder
        #    on the data and some prior, acted as a regularizer
        #    "this can be interpreted as the number of nats required for transmitting the latent space distribution
        #    given the prior"
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean)
                                               + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)

        # combine the reconstruction and latent loss
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 28x28x1 -> 14x14x16
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 14x14x16 -> 7x7x32 [parameter sharing across the 16 depth slices]
            h2_flat = tf.reshape(h2,[self.batchsize, 7*7*32])

            w_mean = dense(h2_flat, 7*7*32, self.n_z, "w_mean")#fully connected
            w_stddev = dense(h2_flat, 7*7*32, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 7*7*32, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 7, 7, 32]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 14, 14, 16], "g_h1"))
            h2 = conv_transpose(h1, [self.batchsize, 28, 28, 1], "g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

    def train(self):
        visualization = self.mnist.train.next_batch(self.batchsize)[0]
        reshaped_vis = visualization.reshape(self.batchsize,28,28)
        ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(10):
                for idx in range(int(self.n_samples / self.batchsize)):
                    batch = self.mnist.train.next_batch(self.batchsize)[0]
                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss),
                                                     feed_dict={self.images: batch})
                    # dumb hack to print cost every epoch
                    if idx % (self.n_samples - 3) == 0:
                        print("epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss)))
                        saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                        generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batchsize,28,28)
                        ims("results/"+str(epoch)+".jpg",merge(generated_test[:64],[8,8]))

class LatentAttention_dropout_z():
    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.n_samples = self.mnist.train.num_examples

        self.n_hidden = 500
        self.n_z = 20
        self.dropout = .1
        self.batchsize = 100

        self.images = tf.placeholder(tf.float32, [None, 784])# 28*28 = 784
        image_matrix = tf.reshape(self.images,[-1, 28, 28, 1])
        z_mean, z_stddev = self.recognition(image_matrix)
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 28*28])

        # define the loss, which includes recalled image errors & gaussian assumptions in variational autoencoder
        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat)
                                              + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev)
                                               - tf.log(tf.square(z_stddev)) - 1,1)

        # self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.cost = tf.reduce_mean(self.generation_loss)#to test the effect of latent_loss
        # 1.if not constrain the latent loss, the latent loss tends to get very large (3000 as opposed to .2)
        # the recalled images seem to be sharp, but the MSE is not as good as when both losses were used (~60 vs. 30)
        # 2.if no latent loss, and have a dropout rate as 0.2
        # the recalled images seem to be more generic, gen_loss ~ 150; latent_loss ~ 75 (only calculated, not in loss func)
        # 4.if no latent loss, dropout rate .1, more generic and overall 4-5 patterns?
        # gen_loss ~ 175; latent_loss ~ 88
        # 3.if no latent loss, and have a dropout rate as 0.01 - so almost nothing remains, the network seems to have 2 structures
        # overall blob & line
        # gen_loss ~ 200; latent_loss ~ 100

        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 28x28x1 -> 14x14x16
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 14x14x16 -> 7x7x32 [parameter sharing across the 16 depth slices]
            h2_flat = tf.reshape(h2,[self.batchsize, 7*7*32])

            w_mean = dense(h2_flat, 7*7*32, self.n_z, "w_mean")#fully connected
            w_stddev = dense(h2_flat, 7*7*32, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            # WM, assume random dropout of sampled z, as if the mean & std were forgotten
            # alternatively, a random value was imputed at the place when z was dropped (imputation of missing data)
            z_dropout = tf.nn.dropout(z, self.dropout, noise_shape=None, seed=None, name=None)

            z_develop = dense(z_dropout, self.n_z, 7*7*32, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 7, 7, 32]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 14, 14, 16], "g_h1"))
            h2 = conv_transpose(h1, [self.batchsize, 28, 28, 1], "g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

    def train(self):
        visualization = self.mnist.train.next_batch(self.batchsize)[0]
        reshaped_vis = visualization.reshape(self.batchsize,28,28)
        ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(10):
                for idx in range(int(self.n_samples / self.batchsize)):
                    batch = self.mnist.train.next_batch(self.batchsize)[0]
                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss),
                                                     feed_dict={self.images: batch})
                    # dumb hack to print cost every epoch
                    if idx % (self.n_samples - 3) == 0:
                        print("epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss)))
                        saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                        generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batchsize,28,28)
                        ims("results/"+str(epoch)+".jpg",merge(generated_test[:64],[8,8]))

class LatentAttention_random_dot():#substitude the dataset with random positioned dots images
    def __init__(self):
        #self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.n_samples = 55000 #arbitrary number of training examples

        self.n_hidden = 500
        self.n_z = 6 #dimension of latent space
        self.batchsize = 100

        self.images = tf.placeholder(tf.float32, [None, 784])# 28*28 = 784
        image_matrix = tf.reshape(self.images,[-1, 28, 28, 1])
        z_mean, z_stddev = self.recognition(image_matrix)
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 28*28])

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat)
                                              + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)
        self.latent_loss = .5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev)
                                               - tf.log(tf.square(z_stddev)) - 1,1) #beta-vae
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 28x28x1 -> 14x14x16
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 14x14x16 -> 7x7x32 [parameter sharing across the 16 depth slices]
            h2_flat = tf.reshape(h2,[self.batchsize, 7*7*32])

            w_mean = dense(h2_flat, 7*7*32, self.n_z, "w_mean")#fully connected
            w_stddev = dense(h2_flat, 7*7*32, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 7*7*32, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 7, 7, 32]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 14, 14, 16], "g_h1"))
            h2 = conv_transpose(h1, [self.batchsize, 28, 28, 1], "g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

    def next_random_dot_batch(self):                  # random dot images generated online
        random_dot_imgs = np.empty([self.batchsize, 28*28])
        H = matlab_style_gauss2D([15, 15], 2)  # lowpass filter

        # random integer between 1 and 10 indicating the set size: how many dots in img
        a = np.ones(self.batchsize, dtype=int)
        # a = np.random.randint(1, 11, self.batchsize)
        for i_img in range(self.batchsize):
            dot_img = np.zeros([28, 28])
            coords = np.random.randint(0, 28, [a[i_img], 2])
            for x, y in coords:
                dot_img[x, y] = 1                     # one pixel dot
            dot_img = scipy.ndimage.convolve(dot_img, H, mode='nearest') # filter
            dot_img = np.reshape(dot_img, [1, 28*28]) # flatten
            dot_img = dot_img / np.max(dot_img)       # normalize to 1
            random_dot_imgs[i_img,:] = dot_img        # combine the whole batch
        return random_dot_imgs

    def train(self):
        np.random.seed(10)
        visualization = self.next_random_dot_batch()
        reshaped_vis = visualization.reshape(self.batchsize,28,28)
        ims("results_random_dot/base.png",merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(10):
                for idx in range(int(self.n_samples / self.batchsize)):
                    #batch = self.mnist.train.next_batch(self.batchsize)[0]
                    # generate random_dot
                    batch = self.next_random_dot_batch()

                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss),
                                                     feed_dict={self.images: batch})
                    # dumb hack to print cost every epoch
                    if idx % (self.n_samples - 3) == 0:
                        print("epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss)))
                        saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                        generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batchsize,28,28)
                        ims("results_random_dot/"+str(epoch)+".png",merge(generated_test[:64],[8,8]))

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# model = LatentAttention()
# model = LatentAttention_dropout_z()
model = LatentAttention_random_dot()
model.train()
