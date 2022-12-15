cd /dhc/home/shahryar.khorasani/NeuroGenomeNet/keras/
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

%env CUDA_DEVICE_ORDER=PCI_BUS_ID
%env CUDA_VISIBLE_DEVICES=0

import os
import glob
import keras.backend as K
import tensorflow as tf
import keras
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop, Adam
from keras.models import Model
from keras.losses import binary_crossentropy

import pandas as pd
import numpy as np

from utils.generator import UKB_DataGenerator
from utils.config_manager import get_directory
from utils.fast_mri_view import *
from utils.losses import generalised_dice_loss_3D, dice_coef_foreground_3D, dice_coef_background_3D
from utils.models import CAE_Reg

import time
import datetime
from scipy.stats import zscore
import random
import pickle
from keras.losses import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras.models import load_model
from sklearn.decomposition import PCA
from scipy import stats

def padwidth(wid):
    wid = np.max((0, wid))
    wid /= 2
    return int(np.ceil(wid)), int(np.floor(wid))


def cropwidth(wid):
    wid = np.min((0, wid))
    wid = np.abs(wid)
    wid /= 2
    return int(np.ceil(wid)), int(np.floor(wid))


def padcrop(img, dim):
    '''
    pads or crops a rescaled scan to given target dimensions x,y,z
    '''
    new_img = np.zeros(dim)
    target_dim = np.array(dim)
    difs = target_dim - np.array(img.shape)
    cropped_img = None
    if np.any(difs < 0):
        crop_x = cropwidth(difs[0])
        crop_y = cropwidth(difs[1])
        crop_z = cropwidth(difs[2])
        cropped_img = img[crop_x[0]:(img.shape[0] - crop_x[1]), crop_y[0]:(img.shape[1] - crop_y[1]),
                      crop_z[0]:(img.shape[2] - crop_z[1])]
        # print(cropped_img.shape)
    else:
        cropped_img = img
    if np.any(difs > 0):
        new_img[:, :, :] = np.pad(cropped_img, (padwidth(difs[0]), padwidth(difs[1]), padwidth(difs[2])),
                                  mode='constant')
    else:
        new_img[:, :, :] = cropped_img
    return new_img

def specif_crop(scan, crop=((0,0),(0,0),(0,0))):
    
    if type(scan) == str: 
        scan = nib.load(scan)
        scan = scan.get_fdata()
        
    x, y, z = scan.shape
    X =np.empty((x, y, z))
    X[:,:,:] = scan
    d1, d2, d3 = (x-crop[0][0]-crop[0][1]), (y-crop[1][0]-crop[1][1]), (z-crop[2][0]-crop[2][1])
    cropped = np.empty((d1, d2, d3))
    cropped[:,:,:] = X[crop[0][0]:(x-crop[0][1]), crop[1][0]:(y-crop[1][1]), crop[2][0]:(z-crop[2][1])]
    return(cropped)

        
class SynthsegDataGenerator(keras.utils.Sequence):
    def __init__(self, ids, regression='Volume_of_white_matter', label=None, y2_dtype='float32', target_dim=(128,128,128), batch_size=1,
                 n_channels=1, dtype='int8', data='ukb',dataframe=None, dataframe_index='Subject_ID', shuffle=False, crop=((0, 0), (0, 0), (0, 0))):
        'Initialization'
        self.ids = ids
        self.data = data
        self.target_dim = target_dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.dtype = dtype
        self.y2_dtype = y2_dtype
        self.dataframe = dataframe
        self.regression = regression
        self.label =label
        self.crop = crop
        self.shuffle = shuffle
        self.dataframe_index=dataframe_index
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ids) / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)    
    
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.ids[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.target_dim, self.n_channels), dtype=self.dtype)
        
        y1 = np.empty((self.batch_size, *self.target_dim, self.n_channels), dtype=self.dtype)
        
        y2 = np.empty((self.batch_size,), dtype=self.y2_dtype)
        y = [y1, y2]

        #data dict
        ddict = dict({'ukb':'/dhc/projects/ukbiobank/original/imaging/brain_mri/T1_structural_brain_mri/unzipped/',
                    'adni':'/dhc/groups/fglippert/adni_t1_mprage/'})
        # Generate data
        
        for i, ID in enumerate(list_IDs_temp):
        
            scan = nib.load(ddict.get(self.data) + ID + '/synthseg/T1_unbiased_brain_synthseg.nii.gz').get_fdata()
            scan = scan.astype(np.int8) 
            scan = np.isin(scan, self.label)
            if self.crop != ((0,0),(0,0),(0,0)):
                scan = specif_crop(scan, crop=self.crop)
            scan = padcrop(scan, self.target_dim)
            scan = scan[ :, :, :, np.newaxis]

            X[i,] = scan
            y1[i] = scan
            
            value = np.array(self.dataframe[self.dataframe[self.dataframe_index]==ID][self.regression])
            value = value.astype(self.y2_dtype)
            y2[i] = value
        return X, y    
    
    
adni = pd.read_csv(get_directory.adni_synthseg)

adni.drop_duplicates(subset='Subject_ID',keep='last',inplace=True)

path = []
for subject in adni.Subject_ID:
    p = glob.glob('/dhc/groups/fglippert/adni_t1_mprage/' + subject + '/*/*/*/synthseg/T1_unbiased_brain_synthseg.nii.gz')
    p = p[-1].split('adni_t1_mprage/')[1]
    p = p.split('/synthseg/')[0]
    path.append(p)
    
adni['scan_dir'] = path

adni['ventricles'] = adni.left_lateral_ventricle + adni.right_lateral_ventricle + adni['3rd_ventricle'] + adni['4th_ventricle'] + adni.left_inferior_lateral_ventricle + adni.right_inferior_lateral_ventricle

adni.insert(2,'z_v_of_ventricles', zscore(adni.ventricles))

AD = adni[adni.Group == 'AD']
CN = adni[adni.Group == 'CN']
MCI = adni[adni.Group == 'MCI']

with open(get_directory.synthseg_dict, 'rb') as handle:
    dict_synthseg = pickle.load(handle)
labels = [dict_synthseg['right_lateral_ventricle'],dict_synthseg['left_lateral_ventricle'],dict_synthseg['3rd_ventricle'],dict_synthseg['4th_ventricle'],dict_synthseg['left_inferior_lateral_ventricle'],dict_synthseg['right_inferior_lateral_ventricle']]

model = load_model('/dhc/home/shahryar.khorasani/models/ukb/regression_auxiliary/2022_10_10_171641/ukb_recon_reg_ventricles.h5')

d_mci = SynthsegDataGenerator(list(MCI.scan_dir), batch_size=10, dataframe=adni, data='adni', dataframe_index='scan_dir', dtype='int8', regression='z_v_of_ventricles', label=labels, 
                                      target_dim=(160, 160, 160), crop=((1,1),(34,34),(2,22))) 

pred_mci = model.predict_generator(d_mci,  workers=1, verbose=1, use_multiprocessing=False)

mci_recon, mci_reg = pred_mci

np.save('/dhc/home/shahryar.khorasani/models/ukb/regression_auxiliary/2022_10_10_171641/mci_reg.npy', mci_reg)
np.save('/dhc/home/shahryar.khorasani/models/ukb/regression_auxiliary/2022_10_10_171641/mci_recon.npy', mci_recon)
