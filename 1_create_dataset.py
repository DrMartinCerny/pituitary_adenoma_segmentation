# Copyright (c) Martin Cerny 2022
# Licensed under Creative Commons Zero v1.0 Universal license
# Not intended for clinical use

import sys
import glob
import os
import numpy as np
import SimpleITK as sitk
import h5py
from sklearn.preprocessing import StandardScaler

from src.Config import Config
from src.ImageRegistration import ImageRegistration

config_file = sys.argv[1]
dataset_source_folder = sys.argv[2]
dataset_target_file = sys.argv[3]

config = Config(config_file)
imageRegistration = ImageRegistration(config)

dataset_target_file = h5py.File(dataset_target_file, "w")

for split in ['train', 'val']:
    files = glob.glob(os.path.join(dataset_source_folder, split, '**'))
    dataset_X = []
    dataset_y = []
    negative_samples = []
    for i, subject in enumerate(files):
        print('{}/{} {}'.format(i+1,len(files),subject))
        mask = os.path.join(subject, 'mask.nii')
        cor_t1_c = os.path.join(subject, 'COR_T1_C.nii')
        cor_t1 = os.path.join(subject, 'COR_T1.nii')
        cor_t2 = os.path.join(subject, 'COR_T2.nii')
        if os.path.exists(mask) and os.path.exists(cor_t1_c):
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask, sitk.sitkInt16))
            if np.sum(mask) > 0:
                # LOAD IMAGES
                cor_t1_c = sitk.ReadImage(cor_t1_c, sitk.sitkFloat32)
                cor_t1 = sitk.ReadImage(cor_t1, sitk.sitkFloat32) if os.path.exists(cor_t1) else None
                cor_t2 = sitk.ReadImage(cor_t2, sitk.sitkFloat32) if os.path.exists(cor_t2) else None

                # REGISTER IMAGES TO T COR C
                cor_t1_transform = imageRegistration.findTransformation(cor_t1_c, cor_t1) if cor_t1 is not None else None
                cor_t2_transform = imageRegistration.findTransformation(cor_t1_c, cor_t2) if cor_t2 is not None else None

                # TRANSFORM IMAGES TO COR T1 COORDINATE SPACE
                cor_t1 = sitk.Resample(cor_t1, cor_t1_c, cor_t1_transform, sitk.sitkBSpline, 0) if cor_t1 is not None else None
                cor_t2 = sitk.Resample(cor_t2, cor_t1_c, cor_t2_transform, sitk.sitkBSpline, 0) if cor_t2 is not None else None

                # GET VOXEL ARRAY DATA
                cor_t1_c = sitk.GetArrayFromImage(cor_t1_c)
                cor_t1 = sitk.GetArrayFromImage(cor_t1) if cor_t1 is not None else None
                cor_t2 = sitk.GetArrayFromImage(cor_t2) if cor_t2 is not None else None
                assert(mask.shape==cor_t1_c.shape)

                # FIND IMAGE CENTER
                centerX = int(cor_t1_c.shape[1]/2)
                centerY = int(cor_t1_c.shape[2]/2)
                top = int(centerY-config.IMG_SIZE_UNCROPPED/2)
                bottom = int(centerY+config.IMG_SIZE_UNCROPPED/2)
                left = int(centerX-config.IMG_SIZE_UNCROPPED/2)
                right = int(centerX+config.IMG_SIZE_UNCROPPED/2)

                # CROP IMAGES
                mask = mask[:,left:right,top:bottom]
                cor_t1_c = cor_t1_c[:,left:right,top:bottom]
                cor_t1 = cor_t1[:,left:right,top:bottom] if cor_t1 is not None else None
                cor_t2 = cor_t2[:,left:right,top:bottom] if cor_t2 is not None else None

                # NORMALIZE CROPPED IMAGES TO ZERO MEAN AND UNIT VARIANCE
                cor_t1_c = StandardScaler().fit_transform(cor_t1_c.flatten().reshape(-1,1)).reshape((len(cor_t1_c),config.IMG_SIZE_UNCROPPED,config.IMG_SIZE_UNCROPPED))
                cor_t1 = StandardScaler().fit_transform(cor_t1.flatten().reshape(-1,1)).reshape((len(cor_t1_c),config.IMG_SIZE_UNCROPPED,config.IMG_SIZE_UNCROPPED)) if cor_t1 is not None else np.zeros(cor_t1_c.shape)
                cor_t2 = StandardScaler().fit_transform(cor_t2.flatten().reshape(-1,1)).reshape((len(cor_t1_c),config.IMG_SIZE_UNCROPPED,config.IMG_SIZE_UNCROPPED)) if cor_t2 is not None else np.zeros(cor_t1_c.shape)
                
                # IDENTIFY SLICES FOR BOTH POSITIVE AND NEGATIVE DATASET
                labeledSlices = np.sum(mask, axis=(1,2)) > 0
                positiveDatasetSlices = [x[0] for x in np.argwhere(labeledSlices)]
                negativeDatasetSlices = [x[0] for x in np.argwhere(np.invert(labeledSlices))]
                if 0 in positiveDatasetSlices: positiveDatasetSlices.remove(0)
                if 0 in negativeDatasetSlices: negativeDatasetSlices.remove(0)
                if len(labeledSlices)-1 in positiveDatasetSlices: positiveDatasetSlices.remove(len(labeledSlices)-1)
                if len(labeledSlices)-1 in negativeDatasetSlices: negativeDatasetSlices.remove(len(labeledSlices)-1)
                negativeDatasetSlices = np.random.permutation(negativeDatasetSlices)[:len(positiveDatasetSlices)]
                
                # ADD TO DATASET
                for slice in positiveDatasetSlices:
                    dataset_y.append(mask[slice])
                    dataset_X.append(np.stack([cor_t1_c[slice-1:slice+2],cor_t1[slice-1:slice+2],cor_t2[slice-1:slice+2]],axis=-1))
                for slice in negativeDatasetSlices:
                    negative_samples.append(np.stack([cor_t1_c[slice-1:slice+2],cor_t1[slice-1:slice+2],cor_t2[slice-1:slice+2]],axis=-1))

    dataset_X = np.stack(dataset_X)
    negative_samples = np.stack(negative_samples)
    dataset_y = np.stack(dataset_y)

    sample_indices = np.random.permutation(np.arange(len(dataset_X)))
    negative_sample_indices = np.random.permutation(np.arange(len(negative_samples)))

    print(dataset_X.shape, dataset_y.shape, negative_samples.shape)

    dataset_target_file.create_dataset("X_"+split, data=dataset_X[sample_indices])
    dataset_target_file.create_dataset("y_"+split, data=dataset_y[sample_indices])
    dataset_target_file.create_dataset("N_"+split, data=negative_samples[negative_sample_indices])

dataset_target_file.close()