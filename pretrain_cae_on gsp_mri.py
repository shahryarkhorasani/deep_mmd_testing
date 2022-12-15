from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import keras
from keras.models import Model, load_model
from keras.callbacks import CSVLogger           
import keras.backend as K
from keras.optimizers import Adam
from random import sample

from config_manager import get_directory
from mri_snp_test import MriSnpTest
from data_io import ScanDataGenerator
from ops import DownConv3D, UpConv3D
from CAE import CAE, sg_user, make_train_test_list

df = pd.read_csv(get_directory.GSP_dataframe)
train, test = make_train_test_list(df, .9)
sg_train, sg_test = sg_user(train_ids=train,test_ids=test, mri_h5=get_directory.GSP_ventricles_h5, dictionary=get_directory.GSP_dict, dataframe_dir=get_directory.GSP_dataframe, input_dim=(96,96,96), zooms=1, data_x='imgdata/scan')

CAE = CAE(96,96,96, summary=True)

opt = Adam(lr=0.0002, clipvalue=1., amsgrad=True)
metr = ['mse']

stemdir='/home/skhorasani/GSP/'

log_dir = stemdir + 'CAE_ventricles.csv'
# dir to write event logs into csv1

checkpoint_dir = stemdir + 'CAE_ventricles.h5'
# dir to write checkpoints into h5

if __name__ == '__main__':
    df = pd.read_csv(get_directory.GSP_dataframe)
    train, test = make_train_test_list(df, .9)
    sg_train, sg_test = sg_user(train_ids=train,test_ids=test, mri_h5=get_directory.GSP_ventricles_h5, dictionary=get_directory.GSP_dict,
dataframe_dir=get_directory.GSP_dataframe, input_dim=(96,96,96), zooms=1, data_x='imgdata/scan')
    
    CAE = CAE(96,96,96, summary=True)
    
    opt = Adam(lr=0.0002, clipvalue=1., amsgrad=True)
    metr = ['mse']
    
    stemdir='/home/skhorasani/GSP/'
    
    log_dir = stemdir + 'CAE_ventricles.csv'
    # dir to write event logs into csv1
    
    checkpoint_dir = stemdir + 'CAE_ventricles.h5'
    # dir to write checkpoints into h5
    
    model = CAE(96, 96, 96,  summary=True)
    model.compile(opt, loss='mean_squared_error', metrics=metr)
    if not os.path.exists(stemdir):
        os.makedirs(stemdir)
    csv_logger = CSVLogger(log_dir, append=True, separator=',')
    
    history = model.fit_generator(sg_train, validation_data=sg_test, epochs=400, verbose=1,
                                  max_queue_size=10, workers=8, use_multiprocessing=True, validation_steps=1, shuffle=False, callbacks=[csv_logger])
    history = history.history
    train_loss = history['loss']
    val_loss_values = history['val_loss']
    with open(log_dir, 'a') as logfile:
        logfile.write('{}\t{}\n'.format(train_loss, val_loss_values))
    if os.path.isfile(checkpoint_dir):
        os.remove(checkpoint_dir)
    model.save(checkpoint_dir)