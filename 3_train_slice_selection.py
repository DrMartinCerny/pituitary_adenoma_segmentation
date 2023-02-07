# Copyright (c) Martin Cerny 2022
# Licensed under Creative Commons Zero v1.0 Universal license
# Not intended for clinical use

import sys
import json
import os
import tensorflow as tf
import h5py
import numpy as np

from src.Config import Config
from src.Model import Model

config_file = sys.argv[1]
dataset_file = sys.argv[2]
model_folder = sys.argv[3]
if not os.path.exists(model_folder): os.mkdir(model_folder)

config = Config(config_file)

dataset_file = h5py.File(dataset_file,'r')
X_train = dataset_file['X_train'][:,:,int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),:config.NUM_CHANNELS]
X_val = dataset_file['X_val'][:,:,int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),:config.NUM_CHANNELS]
N_train = dataset_file['N_train'][:,:,int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),:config.NUM_CHANNELS]
N_val = dataset_file['N_val'][:,:,int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),:config.NUM_CHANNELS]
dataset_file.close()

print(X_train.shape, X_val.shape, N_train.shape, N_val.shape)

y_train = np.concatenate([np.ones(len(X_train)),np.zeros(len(N_train))])
y_val = np.concatenate([np.ones(len(X_val)),np.zeros(len(N_val))])

X_train = np.concatenate([X_train,N_train])
del N_train
X_val = np.concatenate([X_val,N_val])
del N_val

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

model = Model(config).slice_selection_model()
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_folder, 'slice-selection.h5'), save_best_only=True, save_weights_only=True)

print(model.summary())


history = model.fit(x=X_train,y=y_train,validation_data=(X_val,y_val),epochs=config.EPOCHS_CLASSIFIERS,callbacks=[model_checkpoint_callback])

with open(os.path.join(model_folder, 'train-history-slice-selection.json'), 'w') as outfile:
    json.dump(history.history, outfile)