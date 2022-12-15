
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
class for constant directories 
'''

class DataDirectory:
    def __init__(self):
        
        
        #UKB DATA:
        
        self.ukb_dataframe_t1_mri = '/mnt/30T/ukbiobank/original/phenotypes/ukb_brain_mri_t1.csv'
        self.ukb_dir_t1_mri = '/mnt/30T/ukbiobank/original/imaging/brain_mri/T1_structural_brain_mri/unzipped/'
        self.ukb_dict_phenotypes = '/home/Shahryar.Khorasani/NeuroGenomeNet/dict_ukb_mri_phenotype.pickle'
        self.ukb_meta_t1_mri = '/mnt/30T/ukbiobank/original/phenotypes/ukb_t1_mri_meta.csv'
        
        #ADNI DATA:
        
        #all adni mri scans in h5 formated (Image_ID x 256 x 256 x 256)
        self.adni_mprage_h5 = '/mnt/30T/adni/ADNI_MPRAGE_ALL.h5'
        
        #all adni mri ventricles in h5 (Image_ID X 96, 96, 96)
        self.adni_mprage_ventricles_h5 = '/mnt/30T/adni/ADNI_MPRAGE_VENTRICLES.h5'
        
        #dictionary needed to browse ADNI_MPRAGE_ALL.h5 
        self.adni_mprage_dict = '/mnt/30T/adni/ADNI_MPRAGE_ALL_DICT.pickle'
        
        #ADNI_MPRAGE_ALL DataFrame
        self.adni_mprage_dataframe = '/mnt/30T/adni/adni_mprage_all.csv'
        
        #adni_mprage subjects overlapping with bed file subjects (n=707) each having only one image
        self.adni_mprage_snp_dataframe = '/mnt/30T/adni/mprage_snp.csv'
        
        #adni mri scans with freesurfer labels in h5 formated (Image_ID x 256 x 256 x 256)
        #self.adni_mri_h5 = '/mnt/lippert01/mri_backup/adni/adni_mprage/mri_preproc.h5'
        
        #adni mri scans in h5 (Image_ID X 96, 96, 96)
        #self.adni_mri_69_h5 = '/mnt/lippert01/mri_backup/adni/adni_mprage/adni_mri_96.h5'
        
        #adni mri ventricles in h5 (Image_ID X 96, 96, 96)
        #self.adni_ventricles_h5 = '/mnt/lippert01/mri_backup/adni/adni_mprage/adni_ventricles.h5'
        
        #dictionary needed to browse the h5 file with the mri scans
        #self.img_dic = '/mnt/lippert01/mri_backup/adni/adni_mprage/imgid_to_index.pickle'
        
        #adni dataframe, normalized and standardized
        self.adni_dataframe = '/mnt/30T/adni/labels_snp_ready.csv'
        
        #adni bed file, filtered (QC : Missing rate per SNP < 0.1, MAF > 0.05, HWE > 0.001)
        self.adni_bedfile ='/mnt/30T/adni/adni_snp_data/WGS_Omni25_BIN_wo_ConsentsIssues_filtered.bed'
        
        #adni brain mask dataframe: 
        self.adni_brain_mask_dataframe = '/mnt/30T/adni/adni_brain_mask/adni_brain_mask.csv'
        
        #adni brain mask h5 (Mask_ID X 256,256,256)
        self.adni_brain_mask_h5 = '/mnt/30T/adni/adni_brain_mask/adni_brain_mask.h5'
        
        #adni brain mask dictionary for h5:
        self.adni_brain_mask_dict='/mnt/30T/adni/adni_brain_mask/adni_brain_mask_h5_dict.pickle'
    
        
        #GSP DATA:
        
        #GSP filterred dataframe:
        self.GSP_filterred_dataframe = '/mnt/30T/GSP/GSP_filterred_dataframe.csv'
        
        #GSP mri h5 (Image_ID(int) x 232 X 232 x 176):
        self.GSP_mri_h5 = '/mnt/30T/GSP/GSP_resized_scans.h5'
        
        
        
get_directory = DataDirectory()

