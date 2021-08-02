'''
Author: Ambareesh Ravi
Date: Jul 31, 2021
Title: model.py
Description:
    Contains the keras CNN classifier model for Traffic Sign classification
'''

from utils import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

def TrafficSign_Model(image_size = (32, 32), channels = 3, n_classes = 43):
    '''
    Creates the CNN model

    Args:
        image_size - size of the inputs as W,H as <tuple>
        channels - number of input channels as <int>
        n_classes - number of output categories as <int>
        
    Returns:
        the model as <tensorflow.keras.models.Model>
    
    Exception:
        -
    '''
    model = Sequential()
    model.add(Conv2D(32, 3, strides = 2, input_shape = tuple(list(image_size)+[channels])))
    model.add(Conv2D(64, 3, strides = 2))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.25))
    model.add(Conv2D(64, 3, strides = 2))
    model.add(Conv2D(128, 3, strides = 2))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation = "softmax" if n_classes > 1 else "sigmoid"))
    return model