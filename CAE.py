from keras.layers import Input, Dense, Conv3D, Dropout, Flatten, Reshape
from keras.models import Model
from ops import DownConv3D, UpConv3D
import keras.backend as K

import numpy as np
import pickle
from random import sample
from data_io import ScanDataGenerator
from config_manager import get_directory

def CAE(x,y,z,filters=[8,16,32,64,128,256],summary=False):
    input = Input((x,y,z,1))
    layer = Conv3D(filters=filters[0], kernel_size=(3,3,3), activation='relu', padding='SAME', kernel_initializer='he_normal',name='enc0')(input)
    layer = DownConv3D(layer, filters=filters[1],name='enc1')
    layer = Conv3D(filters=filters[2], kernel_size=(3,3,3), activation='relu', padding='SAME', kernel_initializer='he_normal',name='enc3')(layer)
    layer = DownConv3D(layer, filters=filters[3],name='enc3_5')
    layer = DownConv3D(layer, filters=filters[4],name='enc4')
    layer = DownConv3D(layer, filters=filters[5],name='enc5')
    layer = DownConv3D(layer, filters=filters[5],name='enc6')
    layer_shape = K.int_shape(layer)
    layer = Flatten()(layer)
    layer = Dropout(.7)(layer)
    
    layer = Dense(1024, activation='relu')(layer)
    
    layer = Dense(layer_shape[1]*layer_shape[2]*layer_shape[3]*layer_shape[4],activation='relu')(layer)
    layer = Reshape((layer_shape[1],layer_shape[2],layer_shape[3],layer_shape[4]))(layer)

    layer = Conv3D(filters=filters[5], kernel_size=(3,3,3), activation='relu', padding='SAME', kernel_initializer='he_normal',name='dc21')(layer)
    layer = UpConv3D(layer,filters=filters[5],name='dc2')
    layer = UpConv3D(layer,filters=filters[4],name='dc3')
    layer = UpConv3D(layer,filters=filters[3],name='dc4')
    layer = UpConv3D(layer,filters=filters[2],name='dc5')
    layer = Conv3D(filters=filters[1], kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer='he_normal',name='dc22')(layer)
    layer = UpConv3D(layer, filters=filters[0],name='dc6')
    layer = Conv3D(filters=1, kernel_size=(3,3,3), activation='linear', padding='SAME', kernel_initializer='he_normal',name='cd7')(layer)

    model = Model([input],[layer])
    if summary:
        print(model.summary(line_length=140))
    return model


def make_train_test_list(df, ratio, subj_id = 'Subject_ID', img_id = 'Image_ID'):
    train_subj = sample(list(df[subj_id].unique()), int(np.round(len(list(df[subj_id].unique()))*ratio)))
    train = list(df[df[subj_id].isin(train_subj)][img_id])
    test = list(df[~df[img_id].isin(train)][img_id])
    return train, test


def sg_user(train_ids=None,test_ids=None, mri_h5=get_directory.adni_mprage_ventricles_h5, dictionary=get_directory.adni_mprage_dict, dataframe_dir=get_directory.adni_mprage_dataframe, input_dim=(96,96,96), zooms=1, data_x='imgdata/mprage'):
    with open(dictionary, 'rb') as pk:
            imgid = pickle.load(pk)
    sg_train = ScanDataGenerator(mri_h5,img_id=train_ids,dataframe=dataframe_dir
                                 ,input_dim=input_dim
                                 ,img_to_i=imgid, zooms=zooms, data_x=data_x)
    sg_test = ScanDataGenerator(mri_h5,img_id=test_ids,dataframe=dataframe_dir
                                 ,input_dim=input_dim
                                 ,img_to_i=imgid,zooms=zooms, data_x=data_x)
    return sg_train, sg_test