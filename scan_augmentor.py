import os
import warnings

import numpy as np
import scipy.ndimage.interpolation as sni
import scipy.ndimage as scimage



class ScanAugmentor:
    def __init__(self, rotate=True, shift=True, contrast=True, flip=True , flip_prob=.5,rotation_range=[-7,7], rotation_std=4, shift_range=[-10,10], shift_std=4, min_darkness=40, darkness_std=30):
        
        self.rotation=rotate
        self.shift=shift
        self.contrast=contrast
        self.flip=flip
        self.rotate=rotate
        
        self.rotation_std=rotation_std
        self.rotation_range=rotation_range
        self.shift_std=shift_std
        self.shift_range=shift_range
        self.flip_prob=flip_prob
        self.min_darkness = min_darkness
        self.darkness_std=darkness_std
        
    def _flip(self, img):
        return img[::-1,:,:]
    
    def _rotate(self, img, axes=(0, 1)):
        return sni.rotate(img, angle=np.clip(np.random.normal(scale=self.rotation_std, size=1), self.rotation_range[0],self.rotation_range[1]), axes=(np.random.choice(axes), 2), reshape=False, mode='constant')

    def _shift(self, img):
        return scimage.shift(img, np.clip(np.random.normal(scale=self.shift_std , size=3), self.shift_range[0], self.shift_range[1]))
    
    def _contrast(self, img):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vmin, vmax = np.percentile(img, q=(np.absolute(np.clip(np.random.normal(scale=self.darkness_std,size=1), a_min=-self.min_darkness, a_max=self.min_darkness)).item(),100))
            img = np.clip(img, vmin, vmax)  # Clip (limit) the values in img
            img = (img - vmin) / (vmax - vmin)
        return img
    
    def augment(self, scan):
        if self.flip:
            if self.flip_prob >= np.random.rand():
                scan = self._flip(scan)
        if self.rotate:
            scan = self._rotate(scan)
        if self.shift:
            scan = self._shift(scan)
        if self.contrast:
            scan = self._contrast(scan)
            
        return scan