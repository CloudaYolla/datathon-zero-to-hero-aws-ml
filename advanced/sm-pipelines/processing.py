
import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import make_column_transformer

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def print_shape():
    print('Data shape: {}, {} positive examples, {} negative examples')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    parser.add_argument('--dummy-argument', type=str, default="hello somebody")
    args, _ = parser.parse_known_args()
    
    print('Received arguments {}'.format(args))

    input_data_path = os.path.join('/opt/ml/processing/input', 'lorem.csv')
    print('Reading input data from {}'.format(input_data_path))
    
    
    print('Train data shape after preprocessing: {}')    
    train_features_output_path = os.path.join('/opt/ml/processing/train', 'lorem.csv')
    
    print('Saving training features to {}')
