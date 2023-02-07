# Copyright (c) Martin Cerny 2022
# Licensed under Creative Commons Zero v1.0 Universal license
# Not intended for clinical use

import tensorflow as tf
import tensorflow.keras.backend as K
import os

from src.PretrainedModel import unet

class Model:

    def __init__(self, config):
        
        self.config = config
        
    def segmentation_model(self, compile=True):

        inputs, downsampling_stack = self.pretrained_model()

        # Downsampling through the model
        skips = downsampling_stack
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        nr_units = 512
        for skip in skips:
            x = tf.keras.layers.Conv2DTranspose(nr_units, 3, strides=2, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])
            if nr_units > 64:
                nr_units /= 2

        # Final layer for computing logits
        x = tf.keras.layers.Conv2DTranspose(
            filters=self.config.LABEL_CLASSES+1, kernel_size=3, strides=2, padding='same', name='predicted_segmentation'
        )(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        if compile:
            model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[self.dice_coef_total, self.dice_coef_tumor, self.dice_coef_ica, self.dice_coef_normal_gland])
        
        return model
    
    def slice_selection_model(self, compile=True):
        
        inputs, downsampling_stack = self.pretrained_model()        
        slice_relevance = tf.keras.layers.Flatten()(downsampling_stack[-1])
        slice_relevance = tf.keras.layers.Dense(128, name='Dense1')(slice_relevance)
        slice_relevance = tf.keras.layers.LeakyReLU(alpha=0.2, name='LeakyReLU1')(slice_relevance)
        slice_relevance = tf.keras.layers.Dense(32, name='Dense2')(slice_relevance)
        slice_relevance = tf.keras.layers.LeakyReLU(alpha=0.2, name='LeakyReLU2')(slice_relevance)
        slice_relevance = tf.keras.layers.Dense(1, activation='sigmoid', name='slice_relevance')(slice_relevance)
        
        model = tf.keras.Model(inputs=inputs, outputs=slice_relevance)
        if compile:
            model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(),metrics='accuracy')
        return model
    
    def pretrained_model(self):
        
        input_shape = [
            1+self.config.ADJACENT_SLICES*2,
            self.config.IMG_SIZE_PADDED,
            self.config.IMG_SIZE_PADDED,
            self.config.NUM_CHANNELS
        ]
        inputs = tf.keras.layers.Input(shape=input_shape)
        embedding = tf.keras.layers.Conv3D(64,3)(inputs)
        embedding = tf.keras.layers.Reshape([self.config.IMG_SIZE, self.config.IMG_SIZE, 64])(embedding)
        embedding = tf.keras.layers.LeakyReLU(alpha=0.3)(embedding)
        embedding = tf.keras.layers.Dense(32)(embedding)
        embedding = tf.keras.layers.LeakyReLU(alpha=0.3)(embedding)
        
        pretrained_model = unet()
        
        block_1_convolution_1 = pretrained_model.get_layer('conv2d_2')(embedding)
        block_1_activation_1 = pretrained_model.get_layer('activation_2')(block_1_convolution_1)
        block_1_convolution_2 = pretrained_model.get_layer('conv2d_3')(block_1_activation_1)
        block_1_activation_2 = pretrained_model.get_layer('activation_3')(block_1_convolution_2)
        block_1_max_pooling = pretrained_model.get_layer('max_pooling2d_1')(block_1_activation_2)
        block_2_convolution_1 = pretrained_model.get_layer('conv2d_4')(block_1_max_pooling)
        block_2_activation_1 = pretrained_model.get_layer('activation_4')(block_2_convolution_1)
        block_2_convolution_2 = pretrained_model.get_layer('conv2d_5')(block_2_activation_1)
        block_2_activation_2 = pretrained_model.get_layer('activation_5')(block_2_convolution_2)
        block_2_max_pooling = pretrained_model.get_layer('max_pooling2d_2')(block_2_activation_2)
        block_3_convolution_1 = pretrained_model.get_layer('conv2d_6')(block_2_max_pooling)
        block_3_activation_1 = pretrained_model.get_layer('activation_6')(block_3_convolution_1)
        block_3_convolution_2 = pretrained_model.get_layer('conv2d_7')(block_3_activation_1)
        block_3_activation_2 = pretrained_model.get_layer('activation_7')(block_3_convolution_2)
        block_3_max_pooling = pretrained_model.get_layer('max_pooling2d_3')(block_3_activation_2)
        
        outputs = [block_1_max_pooling,block_2_max_pooling,block_3_max_pooling]
        
        if self.config.DOWNSAMPLING_LAYERS < 3 or self.config.DOWNSAMPLING_LAYERS > 6:
            raise Exception('The number of downsampling layers has to be between 3 and 6')
        
        for i in range(self.config.DOWNSAMPLING_LAYERS-3):
            convolution_1 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(outputs[-1])
            batch_norm = tf.keras.layers.BatchNormalization()(convolution_1)
            dropout = tf.keras.layers.Dropout(0.5)(batch_norm)
            activation_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(dropout)
            convolution_2 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(activation_1)
            activation_2 = tf.keras.layers.Activation('relu')(convolution_2)
            max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(activation_2)
            outputs.append(max_pooling)
        
        return inputs, outputs
    
    def dice_coef(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        intersection = K.sum(y_true * y_pred)
        return (2. * intersection + self.config.DICE_COEF_SMOOTH) / (K.sum(y_true) + K.sum(y_pred) + self.config.DICE_COEF_SMOOTH)

    def dice_coef_by_class(self, y_true, y_pred, classId):
        y_pred = K.argmax(y_pred)
        y_pred = K.cast(y_pred == classId, dtype='float32')
        y_true = K.cast(y_true == classId, dtype='float32')
        return self.dice_coef(y_true, y_pred)

    def dice_coef_total(self, y_true, y_pred):
        y_pred = K.argmax(y_pred)
        y_pred = K.one_hot(K.cast(y_pred, dtype='int32'), self.config.LABEL_CLASSES+1)
        y_true = K.one_hot(K.cast(y_true, dtype='int32'), self.config.LABEL_CLASSES+1)
        return self.dice_coef(y_true, y_pred)

    def dice_coef_tumor(self, y_true, y_pred):
        return self.dice_coef_by_class(y_true, y_pred, 1)

    def dice_coef_ica(self, y_true, y_pred):
        return self.dice_coef_by_class(y_true, y_pred, 2)

    def dice_coef_normal_gland(self, y_true, y_pred):
        return self.dice_coef_by_class(y_true, y_pred, 3)