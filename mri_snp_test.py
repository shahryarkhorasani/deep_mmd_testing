from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
from pandas_plink import read_plink
from scipy import stats
import numpy as np
from keras.models import Model, load_model
import pandas as pd
import pickle
from sklearn.decomposition import PCA

from data_io import ScanDataGenerator
from config_manager import get_directory

class MriSnpTest:
    def __init__(self, path_to_model, output_csv, n_feats=10, dataframe=get_directory.adni_mprage_snp_dataframe, snp_data=get_directory.adni_bedfile, 
                 path_to_images=get_directory.adni_mprage_ventricles_h5, 
                 feature_space_layer='dense_1', h5_dict=get_directory.adni_mprage_dict,
                 batch_size = 10, scan_dim=(96,96,96), crops=((0, 0), (0, 0), (0, 0)), zoom=1.0):
        
        self.path_to_model=path_to_model
        self.output_csv=output_csv
        self.n_feats=n_feats
        self.dataframe=pd.read_csv(dataframe)
        self.dataframe_path=dataframe
        self.snp_data=snp_data
        self.path_to_images=path_to_images
        self.feature_space_layer=feature_space_layer
        self.h5_dict=h5_dict
        self.batch_size=batch_size
        self.scan_dim=scan_dim
        self.crops=crops
        self.zoom=zoom
        self.snp_data=snp_data

        
    def subject2index(self, subjects, dictionary):
        ids2ind = np.array([dictionary[id] for id in subjects])
        return ids2ind
    
    def extract_features_mri(self):
        # feature extraction
        
        #load model
        model = load_model(self.path_to_model)
        #remove output layer
        new_model = Model(model.inputs, model.get_layer(self.feature_space_layer).output)
        #load the weights
        new_model.set_weights(model.get_weights())

        # loading images using the dictionary to find the scan given an Image_ID:
        with open(self.h5_dict, 'rb') as pk:
            imgid = pickle.load(pk)
        img = ScanDataGenerator(self.path_to_images, img_id=self.dataframe.Image_ID, dataframe=self.dataframe_path, crop=self.crops ,input_dim=self.scan_dim, img_to_i=imgid,zooms=self.zoom)
        np_img = np.array(img)
        np_img = np_img[:,0,0,:,:,:,:]
        output = new_model.predict(np_img[:len(self.dataframe.Image_ID)], batch_size=self.batch_size, verbose=True)

        feats = np.array(output)

        # end feature extraction
        return feats
    
    def compute_p_value(self, feats_X, feats_Y):
        n, d = feats_X.shape
        m = len(feats_Y)
        pca = PCA(n_components=self.n_feats)
        feats = np.vstack((feats_X, feats_Y))
        feats = pca.fit_transform(feats)
        feats_X, feats_Y = feats[:n], feats[n:]
        d = self.n_feats
        mean_fX = feats_X.mean(0)
        mean_fY = feats_Y.mean(0)
        D = mean_fX - mean_fY
    
        eps_ridge = 1e-8    # add ridge to Covariance for numerical stability
        all_features = np.concatenate([feats_X, feats_Y])
        Cov_D = (1./n + 1./m) * np.cov(all_features.T) + eps_ridge * np.eye(d)
    
        statistic = D.dot(np.linalg.solve(Cov_D, D))
        p_val = 1. - stats.chi2.cdf(statistic, d)
        return p_val
    
    def test(self):
        (bim, fam, bed) = read_plink(self.snp_data)
        dataframe=self.dataframe
        mri_subjects = np.array(fam[fam.iid.isin(dataframe.Subject_ID)].index)
        rs_snps = bim[~bim.snp.str.contains(',')][bim.snp.str.contains('rs')].snp
        dictionary = dict(zip(dataframe.Subject_ID,dataframe.index))
        
        all_features = self.extract_features_mri()
        
        with open(self.output_csv, 'w') as csvfile:
            fieldnames = ['snp', 'p_value', 'test_condition', 'x_sample_size', 'y_sample_size', 'na_numbers']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, snp in enumerate(bim.snp[:]):
                if snp in list(rs_snps[184778:]):
                    na_subs = np.argwhere(np.isnan(bed[i][mri_subjects].compute()))
                    na_subject_IDs = fam[fam.index.isin(list(mri_subjects[na_subs]))].iid

                    subs_2 = np.argwhere(bed[i][mri_subjects].compute()==2)
                    AA = fam[fam.index.isin(list(mri_subjects[subs_2]))].iid

                    subs_1 = np.argwhere(bed[i][mri_subjects].compute()==1)
                    Aa = fam[fam.index.isin(list(mri_subjects[subs_1]))].iid

                    subs_0 = np.argwhere(bed[i][mri_subjects].compute()==0)
                    aa = fam[fam.index.isin(list(mri_subjects[subs_0]))].iid

                    #genotype conditions:

                    if (len(AA) > len(Aa)) | (len(AA) >= len(aa)):
                        if len(Aa) > len(aa):
                            X_ids = self.subject2index(AA, dictionary)
                            Y_ids = self.subject2index(Aa, dictionary)
                            condition='AA_vs_Aa'
                        else:
                            X_ids = self.subject2index(AA, dictionary)
                            Y_ids = self.subject2index(aa, dictionary)
                            condition='AA_vs_aa'

                    elif len(Aa) >= len(AA):
                        if len(AA) > len(aa):
                            X_ids = self.subject2index(Aa, dictionary)
                            Y_ids = self.subject2index(AA, dictionary)
                            condition='Aa_vs_AA'
                        else:
                            X_ids = self.subject2index(Aa, dictionary)
                            Y_ids = self.subject2index(aa, dictionary)
                            condition='Aa_vs_aa'

                    elif (len(aa) >= len(Aa)) | (len(aa) >= len(AA)):
                        if len(AA) > len(Aa):
                            X_ids = self.subject2index(aa, dictionary)
                            Y_ids = self.subject2index(AA, dictionary)
                            condition='aa_vs_AA'
                        else:
                            X_ids = self.subject2index(aa, dictionary)
                            Y_ids = self.subject2index(Aa, dictionary)
                            condition='aa_vs_Aa'
                                                
                    x_feat, y_feat = all_features[X_ids], all_features[Y_ids]
                    
                    p = self.compute_p_value(x_feat, y_feat)
                    writer.writerow({'snp':snp, 'p_value':p, 'test_condition':condition, 'x_sample_size':len(X_ids), 'y_sample_size':len(Y_ids), 'na_numbers':len(na_subject_IDs)})