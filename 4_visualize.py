# Copyright (c) Martin Cerny 2022
# Licensed under Creative Commons Zero v1.0 Universal license
# Not intended for clinical use

import sys
import numpy as np
import h5py
import cv2
import os

from src.Config import Config
from src.Visualization import Visualization
from src.Model import Model

config = Config(sys.argv[1])
model = Model(config)

dataset_file = sys.argv[2]
model_folder = sys.argv[3]

dataset_file = h5py.File(dataset_file,'r')

X_val = dataset_file['X_val'][:,:,int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),:config.NUM_CHANNELS].astype(np.float64)
maskGroundTruth = dataset_file['y_val'][:,int(config.CROP_OFFSET/2)+1:-int(config.CROP_OFFSET/2)-1,int(config.CROP_OFFSET/2)+1:-int(config.CROP_OFFSET/2)-1]
dataset_file.close()

model = Model(config).segmentation_model(compile=False)
model.load_weights(os.path.join(model_folder, 'segmentation.h5'))
maskPredicted = np.argmax(model.predict(X_val),axis=-1)

for sample in np.random.randint(0, len(X_val), size=10):
    img = Visualization.toBitmap(X_val[sample,1,1:-1,1:-1,0])
    Visualization.overlay(img,maskPredicted[sample])
    img = Visualization.upsample(img)
    Visualization.contours(img,maskGroundTruth[sample])
    cv2.imshow('Segmentation visualization', img)
    cv2.waitKey(0)