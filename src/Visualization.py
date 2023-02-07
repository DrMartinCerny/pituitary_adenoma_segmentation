# Copyright (c) Martin Cerny 2022
# Licensed under Creative Commons Zero v1.0 Universal license
# Not intended for clinical use

import numpy as np
import cv2

class Visualization:

    def toBitmap(img,intensity_range=2):
        # transforms image from dataset to RGB bitmap
        img = np.clip(img,-intensity_range,intensity_range)
        img = ((img+intensity_range)/(intensity_range*2))
        return np.stack([img,img,img],axis=-1)
    
    def overlay(img,mask,a=0.7):
        # overlay with colored areas for labels
        img[mask==1,0] *= a
        img[mask==1,2] *= a
        img[mask==2,2] += a/2
        img[mask==3,2] *= a
        img[mask==3,1] *= a
        img = np.clip(img, 0, 1)
    
    def contours(img,mask,upsampling=2):
        # draw contours for individual labels
        cv2.drawContours(img, cv2.findContours(Visualization.upsample((mask==1).astype(np.uint8),upsampling=upsampling), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0,255,0), 1, cv2.LINE_AA)
        cv2.drawContours(img, cv2.findContours(Visualization.upsample((mask==2).astype(np.uint8),upsampling=upsampling), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0,0,255), 1, cv2.LINE_AA)
        cv2.drawContours(img, cv2.findContours(Visualization.upsample((mask==3).astype(np.uint8),upsampling=upsampling), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255,0,0), 1, cv2.LINE_AA)
    
    def upsample(img,upsampling=2):
        for i in range(upsampling-1):
            img = cv2.pyrUp(img)
        return img