from mri_3d import compute_p_value, extract_features_mri, compute_p_value_mmd
from config_manager import get_directory
import numpy as np
import math
import pandas as pd
adni = pd.read_csv(get_directory.adni_mprage_dataframe)
tadpole = pd.read_csv('/mnt/lippert01/mri_backup/adni/adni_mprage/TADPOLE_D1_D2.csv', low_memory=False)
tadpole.rename(columns={'PTID':'Subject_ID'},inplace=True)
data_apoe4 = tadpole[['Subject_ID','APOE4']]
ADNI = pd.merge(adni, data_apoe4, how='left', on='Subject_ID')
ADNI.drop_duplicates(subset='Subject_ID',inplace=True)

def get_n_features(x,y):
    n = np.int(np.round(math.sqrt((x.shape[0]+y.shape[0])/2)))
    return n 

x_normal, y_ad = extract_features_mri(path_to_model='/home/skhorasani/GSP/CAE_ventricles.h5',X_ids=ADNI[ADNI.DX_Group == 'Normal'].Image_ID, Y_ids=ADNI[ADNI.DX_Group == 'AD'].Image_ID)
p_normal_ad = compute_p_value(x_normal, y_ad, n_feats=get_n_features(x_normal, y_ad))
print('x_normal', 'y_ad', p_normal_ad)

x_normal, y_mci = extract_features_mri(path_to_model='/home/skhorasani/GSP/CAE_ventricles.h5',X_ids=ADNI[ADNI.DX_Group == 'Normal'].Image_ID, Y_ids=ADNI[ADNI.DX_Group == 'MCI'].Image_ID)
p_normal_mci = compute_p_value(x_normal, y_mci, n_feats=get_n_features(x_normal, y_mci))
print('x_normal', 'y_mci', p_normal_mci)

x_mci, y_ad = extract_features_mri(path_to_model='/home/skhorasani/GSP/CAE_ventricles.h5',X_ids=ADNI[ADNI.DX_Group == 'MCI'].Image_ID, Y_ids=ADNI[ADNI.DX_Group == 'AD'].Image_ID)
p_mci_ad = compute_p_value(x_mci, y_ad, n_feats=get_n_features(x_mci, y_ad))
print('x_mci', 'y_ad', p_mci_ad)

x_apoe4_0, y_apoe4_1 = extract_features_mri(path_to_model='/home/skhorasani/GSP/CAE_ventricles.h5',X_ids=ADNI[ADNI.APOE4 == 0].Image_ID, Y_ids=ADNI[ADNI.APOE4 == 1].Image_ID)
p_apoe4_0_1 = compute_p_value(x_apoe4_0, y_apoe4_1, n_feats=get_n_features(x_apoe4_0, y_apoe4_1))
print('x_apoe4_0', 'y_apoe4_1', p_apoe4_0_1)