# Copyright (c) Martin Cerny 2022
# Licensed under Creative Commons Zero v1.0 Universal license
# Not intended for clinical use

import wget
import zipfile
import os

response = wget.download('https://storage.googleapis.com/example-dataset-bucket/example-dataset.zip', 'data/example-dataset.zip')
with zipfile.ZipFile('data/example-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('data')
os.remove('data/example-dataset.zip')