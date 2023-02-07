# Copyright (c) Martin Cerny 2022
# Licensed under Creative Commons Zero v1.0 Universal license
# Not intended for clinical use

import yaml

class Config:

    def __init__(self, config_file):
        with open(config_file, 'r') as config_file:
            config = yaml.safe_load(config_file)
            
            self.IMG_SIZE = config['DATASET_EXTRACTION']['IMG_SIZE']
            self.NUM_CHANNELS = config['DATASET_EXTRACTION']['NUM_CHANNELS']
            self.ADJACENT_SLICES = config['DATASET_EXTRACTION']['ADJACENT_SLICES']
            self.CROP_OFFSET = config['DATASET_EXTRACTION']['CROP_OFFSET']
            self.IMG_SIZE_PADDED = self.IMG_SIZE + self.ADJACENT_SLICES*2
            self.IMG_SIZE_UNCROPPED = self.IMG_SIZE_PADDED + self.CROP_OFFSET
            self.LABEL_CLASSES = config['DATASET_EXTRACTION']['LABEL_CLASSES']
            self.IMAGE_REGISTRATION_EPOCHS = config['DATASET_EXTRACTION']['IMAGE_REGISTRATION_EPOCHS']
            
            self.DOWNSAMPLING_LAYERS = config['MODEL_CONFIGURATION']['DOWNSAMPLING_LAYERS']
            self.DICE_COEF_SMOOTH = config['MODEL_CONFIGURATION']['DICE_COEF_SMOOTH']
            self.ONLY_NONNULL_INPUTS = config['MODEL_CONFIGURATION']['ONLY_NONNULL_INPUTS']
            
            self.BATCH_SIZE = config['TRAINING']['BATCH_SIZE']
            self.EPOCHS_SEGMENTATION = config['TRAINING']['EPOCHS_SEGMENTATION']
            self.EPOCHS_CLASSIFIERS = config['TRAINING']['EPOCHS_CLASSIFIERS']