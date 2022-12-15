import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def show_slices(scan, x=None, y=None, z=None, size=30): #scan: directory to a nifti file
    """ Function to display the middle slice in each axis of a scan """
    fig, axes = plt.subplots(1, 3, figsize=(size,size))
    if type(scan) == str: 
        scan = nib.load(scan)
        scan = scan.get_fdata()
    elif type(scan) == np.array:
        scan = scan
    if len(scan.shape) == 3:
        if x:
            slice_x = scan[x,:,:]
        else:    
            slice_x = scan[int(scan.shape[0]/2),:,:]
        if y:
            slice_y = scan[:,y,:]
        else:
            slice_y = scan[:,int(scan.shape[1]/2),:]
        if z:
            slice_z = scan[:,:,z]
        else:
            slice_z = scan[:,:,int(scan.shape[2]/2)]
        slices = [slice_x, slice_y, slice_z]
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, origin="lower")
    elif len(scan.shape) == 4: 
        if x:
            slice_x = scan[x,:,:, 0]
        else:    
            slice_x = scan[int(scan.shape[0]/2),:,:, 0]
        if y:
            slice_y = scan[:,y,:, 0]
        else:
            slice_y = scan[:,int(scan.shape[1]/2),:, 0]
        if z:
            slice_z = scan[:,:,z, 0]
        else:
            slice_z = scan[:,:,int(scan.shape[2]/2), 0]
        slices = [slice_x, slice_y, slice_z]
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, origin="lower")