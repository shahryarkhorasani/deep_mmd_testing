import os
import warnings
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
import pandas as pd
import pickle

from scan_augmentor import ScanAugmentor


def ScanDataConventor(input_h5=None, input_h5_keys='imgdata/mprage', input_h5_key_ids='labels/img_id', crop = ((32, 32),(22, 42), (64, 0)), zooms=None, dtype='f8',augment=False, output_h5=None):
    '''
    function to crop and scale images from an h5 and save them to an h5
    '''
    with h5py.File(input_h5, 'r') as r:
        shape = r[input_h5_keys].shape
        img_ids = (r[input_h5_key_ids][:][:]).tolist()
    n, x, y, z = shape[0], shape[1], shape[2], shape[3] 
    d1, d2, d3 = (x-crop[0][0]-crop[0][1]), (y-crop[1][0]-crop[1][1]), (z-crop[2][0]-crop[2][1])
    shape_cropped = (n, d1, d2, d3)
    shape_zoomed = (n, int(np.round(shape_cropped[1]*zooms)),int(np.round(shape_cropped[2]*zooms)),int(np.round(shape_cropped[3]*zooms)))
    
    with h5py.File(output_h5, 'w') as w:
        w.create_group(input_h5_key_ids.split('/')[0])
        w.create_group(input_h5_keys.split('/')[0])
        labels = w.create_dataset(input_h5_key_ids, maxshape = (None,None), shape=(n, 1), dtype=int , data=img_ids ,compression="lzf")
        out_scan = w.create_dataset(input_h5_keys, maxshape = (None,None,None,None), shape=(n, shape_zoomed[1], shape_zoomed[2], shape_zoomed[3]), dtype=dtype, 
                                        chunks=(1, shape_zoomed[1], shape_zoomed[2], shape_zoomed[3]) ,compression="lzf")
    
        for scan in range(len(range(n))):
            with h5py.File(input_h5, 'r') as r:
                scan_data = r[input_h5_keys][scan, :, :, :]
                r.close()
            if augment:
                scan_data = ScanAugmentor().augment(scan_data)
            if crop != ((0,0),(0,0),(0,0)): 
                scan_cropped = np.empty((d1, d2, d3))
                scan_cropped[:,:,:] = scan_data[crop[0][0]:(x-crop[0][1]), crop[1][0]:(y-crop[1][1]), crop[2][0]:(z-crop[2][1])]
            else:
                scan_cropped = scan_data
            
            if zooms != 1:            
                scan_zoom = zoom(scan_cropped[:, :, :], zooms)
            else:
                scan_zoom = scan_cropped
    
            out_scan[scan, :, :, :] = scan_zoom
            if scan % 200==0:
                print('{} out of {} processed.'.format(scan, n))
        w.flush()
        w.close()