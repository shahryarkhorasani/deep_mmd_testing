from scipy import stats
import numpy as np
from keras.models import Model, load_model
import pandas as pd
import pickle
import h5py
from sklearn.decomposition import PCA

from data_io import ScanDataGenerator
from config_manager import get_directory

def extract_features_mri(path_to_images='/mnt/30T/adni/ADNI_MPRAGE_VENTRICLES.h5', path_to_model=None, feature_space_layer='dense_1', h5_dict=get_directory.adni_mprage_dict, X_ids=None, Y_ids=None, batch_sizes = 10, scan_dim=(96,96,96), crops=((0, 0), (0, 0), (0, 0)), zoom=1.0):
    # feature extraction
    
    #load model
    model = load_model(path_to_model)
    #remove output layer
    new_model = Model(model.inputs, model.get_layer(feature_space_layer).output)
    #load the weights
    new_model.set_weights(model.get_weights())
    
    # loading images using the dictionary to find the scan given an Image_ID:
    with open(h5_dict, 'rb') as pk:
        imgid = pickle.load(pk)
        
    img = ScanDataGenerator(path_to_images,img_id=X_ids,dataframe=get_directory.adni_dataframe, crop=crops
                                 ,input_dim=scan_dim
                                 ,img_to_i=imgid,zooms=zoom)
    np_img = np.array(img)
    np_img = np_img[:,0,0,:,:,:,:]
    #np_img = np_img.reshape((len(X_ids), scan_dim[0],scan_dim[1],scan_dim[2], 1))
    output_X = new_model.predict(np_img[:len(X_ids)], batch_size=batch_sizes, verbose=True) #images should have the dimensions: (batch_size,scan_dim[0],scan_dim[1],scan_dim[2],1)
    
    img = ScanDataGenerator(path_to_images,img_id=Y_ids,dataframe=get_directory.adni_dataframe, crop=crops
                                 ,input_dim=scan_dim
                                 ,img_to_i=imgid,zooms=zoom)
    np_img = np.array(img)
    np_img = np_img[:,0,0,:,:,:,:]
    #np_img = np_img.reshape((len(Y_ids), scan_dim[0],scan_dim[1],scan_dim[2], 1)) 
    output_Y = new_model.predict(np_img[:len(Y_ids)], batch_size=batch_sizes, verbose=True) #images should have the dimensions: (batch_size,scan_dim[0],scan_dim[1],scan_dim[2],1)
    
    feats_X = np.array(output_X)
    feats_Y = np.array(output_Y)

    # end feature extraction
    return feats_X.astype(np.float128), feats_Y.astype(np.float128)

def compute_p_value(feats_X, feats_Y, n_feats=10):
    n, d = feats_X.shape
    m = len(feats_Y)
    pca = PCA(n_components=n_feats)
    feats = np.vstack((feats_X, feats_Y))
    feats = pca.fit_transform(feats)
    feats_X, feats_Y = feats[:n], feats[n:]
    d = n_feats
    mean_fX = feats_X.mean(0)
    mean_fY = feats_Y.mean(0)
    D = mean_fX - mean_fY

    eps_ridge = 1e-8    # add ridge to Covariance for numerical stability
    all_features = np.concatenate([feats_X, feats_Y])
    Cov_D = (1./n + 1./m) * np.cov(all_features.T) + eps_ridge * np.eye(d)

    statistic = D.dot(np.linalg.solve(Cov_D, D))
    p_val = 1. - stats.chi2.cdf(statistic, d)
    return p_val

def compute_p_value_mmd(feats_X, feats_Y, n_permute=10000):
    stat = compute_mmd(feats_X, feats_Y)
    n, m = len(feats_X), len(feats_Y)
    l = n + m
    feats_Z = np.vstack((feats_X, feats_Y))

    resampled_vals = np.empty(n_permute)
    for i in range(n_permute):
        index = np.random.permutation(l)
        feats_X, feats_Y = feats_Z[index[:n]], feats_Z[index[n:]]
        resampled_vals[i] = compute_mmd(feats_X, feats_Y)
    p_val = np.mean(stat < resampled_vals)
    return p_val
def compute_mmd(X, Y):
    mean_X = X.mean(0)
    mean_Y = Y.mean(0)
    D = mean_X - mean_Y
    stat = np.linalg.norm(D)**2
    return stat
