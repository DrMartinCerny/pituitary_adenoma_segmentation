# Copyright (c) Martin Cerny 2022
# Licensed under Creative Commons Zero v1.0 Universal license
# Not intended for clinical use

import sys
import json
import os
import tensorflow as tf

from src.Config import Config
from src.Generator import Generator
from src.Model import Model

config_file = sys.argv[1]
dataset_file = sys.argv[2]
model_folder = sys.argv[3]
if not os.path.exists(model_folder): os.mkdir(model_folder)

config = Config(config_file)
generator = Generator(dataset_file,True,config)
val_generator = Generator(dataset_file,False,config)
model = Model(config).segmentation_model()
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_folder, 'segmentation.h5'), save_best_only=True, save_weights_only=True)

print(model.summary())

history = model.fit(generator,validation_data=val_generator,epochs=config.EPOCHS_SEGMENTATION,callbacks=[model_checkpoint_callback])

with open(os.path.join(model_folder, 'train-history-segmentation.json'), 'w') as outfile:
    json.dump(history.history, outfile)