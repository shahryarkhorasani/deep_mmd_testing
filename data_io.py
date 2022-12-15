# accesses the scan saved as an h5file using the img_id:
import os
import warnings

import h5py
import keras
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import zoom


class ScanDataGenerator(keras.utils.Sequence):
    '''
    Class for on-the-fly loading and (augmentation) of whole MRI scans from ADNI data set
    TODO: Check compatibility with other data sets
    '''

    def __init__(self, h5file, batch_size=1, img_id=None, data_x='imgdata/mprage', image_id='Image_ID', dataframe=None,
                 label=None, shuffle=False,
                 crop=((0, 0), (0, 0), (0, 0)), input_dim=(256, 256, 256), img_to_i=None, dtype='f8',
                 zooms=1):

        self.h5file = h5file
        self.batch_size = batch_size
        self.data_x = data_x
        self.shuffle = shuffle
        self.crop = crop
        self.input_dim = input_dim
        self.img_ids = img_id
        self.img_to_i = img_to_i  # dictionary mapping image_id to index in h5 file
        self.i = np.array([self.img_to_i[id] for id in img_id])
        self.label = label
        self.image_id = image_id  # get from dataframe
        self.dtype = dtype
        self.dataframe = pd.read_csv(dataframe)
        self.dataframe = self.dataframe.set_index(self.image_id)
        self.on_epoch_end()
        self.zooms = zooms

    def __len__(self):
        return len(self.index) // self.batch_size

    def on_epoch_end(self):
        index = np.arange(len(self.i))
        if self.shuffle:
            np.random.shuffle(index)
        self.index = index

    def __getitem__(self, index):
        INDEX = self.index[index * self.batch_size: (index + 1) * self.batch_size]
        indexes_h5 = self.i[INDEX]
        X = self.__data_generation(indexes_h5)
        if self.label == True:
            Y = np.array(self.dataframe.iloc[INDEX][self.label]) #check if this is working right
        else:
            Y = X
        return X, Y

    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, self.input_dim[0], self.input_dim[1], self.input_dim[2]))
        with h5py.File(self.h5file, 'r') as f:
            for b in range(self.batch_size):
                X[b, :, :, :] = f[self.data_x][indexes[b], :, :, :]
        X = X.astype(self.dtype)
        X_cropped = None
        for scan in range(X.shape[0]):
            X[scan] /= np.max(X[scan])

        if self.crop != ((0,0),(0,0),(0,0)):
            x, y, z = self.input_dim
            d1, d2, d3 = (x-self.crop[0][0]-self.crop[0][1]), (y-self.crop[1][0]-self.crop[1][1]), (z-self.crop[2][0]-self.crop[2][1])
            X_cropped = np.empty((X.shape[0], d1, d2, d3))
            X_cropped[:,:,:,:] = X[:, self.crop[0][0]:(x-self.crop[0][1]), self.crop[1][0]:(y-self.crop[1][1]), self.crop[2][0]:(z-self.crop[2][1])]
        else:
            X_cropped = X
        X_cropped = X_cropped[ :, :, :, :, np.newaxis]

        if self.zooms != 1:
            X_zoom = np.empty((self.batch_size, int(np.round(X_cropped.shape[1] * self.zooms)),
                               int(np.round(X_cropped.shape[2] * self.zooms)),
                               int(np.round(X_cropped.shape[3] * self.zooms)), 1))

            for b in range(self.batch_size):
                X_zoom[b, :, :, :, 0] = zoom(X_cropped[b, :, :, :, 0], self.zooms)

            return X_zoom

        return X_cropped        