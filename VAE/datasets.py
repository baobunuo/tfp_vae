import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imresize
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

class MNISTDataset:
    def __init__(self, batch_size, img_height, img_width):
        self.dataset_name = 'MNIST'
        self.info_url = 'http://yann.lecun.com/exdb/mnist/'

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        self.x_train = np.round(x_train)
        self.x_test = np.round(x_test)

        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = 1
        self.img_pixel_range = (0.0, 1.0)

        self.batch_size = batch_size

        self.train_dataset_size = x_train.shape[0]
        self.train_idxs = np.random.permutation(self.train_dataset_size)
        self.train_idx_offset = 0

        self.test_dataset_size = x_test.shape[0]
        self.test_idxs = np.random.permutation(self.test_dataset_size)
        self.test_idx_offset = 0

    def get_batch(self):
        batch = []
        for b in range(0, self.batch_size):
            idx = self.train_idxs[self.train_idx_offset + b]
            img = self.x_train[idx]
            img = imresize(arr=img, size=[self.img_height, self.img_width]) / 255.0
            img = np.expand_dims(img, 2)
            batch.append(img)

        self.train_idx_offset += self.batch_size

        return np.array(batch)

    def new_epoch(self):
        self.train_idxs = np.random.permutation(self.train_dataset_size)
        self.train_idx_offset = 0

    def get_test_batch(self):
        batch = []
        for b in range(0, self.batch_size):
            idx = self.test_idxs[self.test_idx_offset + b]
            img = self.x_train[idx]
            img = imresize(arr=img, size=[self.img_height, self.img_width]) / 255.0
            img = np.expand_dims(img, 2)
            batch.append(img)

        self.test_idx_offset += self.batch_size

        return np.array(batch)

    def get_reference_batch(self):
        batch = []
        for b in range(0, self.batch_size):
            img = self.x_train[b]
            img = imresize(arr=img, size=[self.img_height, self.img_width]) / 255.0
            img = np.expand_dims(img, 2)
            batch.append(img)
        batch = np.array(batch)

        return batch



class CelebADataset:
    def __init__(self, batch_size, img_height, img_width):
        self.dataset_name = 'CelebA'
        self.info_url = 'http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html'

        self.train_path_suffix = None
        self.test_path_suffix = None

        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = 3
        self.img_pixel_range = (-1.0, 1.0)

        self.batch_size = batch_size

        self.train_path = '/Users/lucaslingle/git/spherical_infogan/data/img_align_celeba/'
        self.test_path = '/Users/lucaslingle/git/spherical_infogan/data/img_align_celeba/'
        self.train_dataset_size = 202599
        self.test_dataset_size = 202599
        self.train_idxs = np.random.permutation(self.train_dataset_size)
        self.test_idxs = np.arange(0, self.test_dataset_size)
        self.train_idx_offset = 0
        self.test_idx_offset = 0

        train_files = os.listdir(self.train_path)
        self.train_files = [os.path.join(self.train_path, fn) for fn in train_files]

        test_files = os.listdir(self.test_path)
        self.test_files = [os.path.join(self.test_path, fn) for fn in test_files]

    def get_batch(self):
        batch = []
        for b in range(0, self.batch_size):
            idx = self.train_idxs[self.train_idx_offset + b]
            img = plt.imread(self.train_files[idx])
            ## crop
            _i = (img.shape[0] // 2) - (108 // 2)
            _j = (img.shape[1] // 2) - (108 // 2)
            img = img[_i:(_i + 108), _j:(_j+108)]
            ## resize
            img = imresize(arr=img, size=[self.img_height, self.img_width])
            ## convert to 0-1 range
            img = img / 256.0
            ## add random noise
            u = np.random.uniform(low=0.0, high=(1.0 / 256.0), size=img.shape)
            img = img + u
            ## convert to (-1., 1.) range for color images
            img = 2.0 * img - 1.0
            batch.append(img)

        self.train_idx_offset += self.batch_size

        return np.array(batch)

    def new_epoch(self):
        self.train_idxs = np.random.permutation(self.train_dataset_size)
        self.train_idx_offset = 0

    def get_test_batch(self):
        batch = []
        for b in range(0, self.batch_size):
            idx = self.test_idxs[self.test_idx_offset + b]
            img = plt.imread(self.test_files[idx])
            ## 0-1 range
            img = img / 255.0
            ## crop
            _i = (img.shape[0] // 2) - (108 // 2)
            _j = (img.shape[1] // 2) - (108 // 2)
            img = img[_i:(_i + 108), _j:(_j+108)]
            ## resize
            img = imresize(arr=img, size=[self.img_height, self.img_width])
            ## scipy.misc.imresize puts everything back on 0-255 scale, so we have to rescale again
            img = img / 255.0
            ## test set: don't add random noise
                        
            ## using (-1., 1.) range for color images
            img = 2.0 * img - 1.0
            batch.append(img)

        self.test_idx_offset += self.batch_size

        return np.array(batch)

    def get_reference_batch(self):
        batch = []
        for b in range(0, self.batch_size):
            img = plt.imread(self.train_files[b])
            ## crop
            _i = (img.shape[0] // 2) - (108 // 2)
            _j = (img.shape[1] // 2) - (108 // 2)
            img = img[_i:(_i + 108), _j:(_j + 108)]
            ## resize
            img = imresize(arr=img, size=[self.img_height, self.img_width])
            ## convert to 0-1 range
            img = img / 255.0
            ## convert to (-1., 1.) range for color images
            img = 2.0 * img - 1.0
            batch.append(img)

        batch = np.array(batch)

        return batch



def get_dataset(name, batch_size, img_height, img_width):
    if name.lower() == 'mnist':
        return MNISTDataset(batch_size, img_height, img_width)
    elif name.lower() == 'celeba':
        return CelebADataset(batch_size, img_height, img_width)
