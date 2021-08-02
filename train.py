'''
Author: Ambareesh Ravi
Date: Jul 31, 2021
Title: train.py
Description:
    Contains the Trainer class for training the Traffic Sign classification model in keras
'''

from model import *
from data import *

class TrafficSign_Trainer:
    # Trains the Traffic Sign classifier model
    def __init__(self,):
        '''
        Initializes the class

        Args:
            -

        Returns:
            -

        Exception:
            -
        '''
        pass
    
    def train(self, model, X, y, model_path = "models/Sign_model.h5", epochs = 10, batch_size = 64, lr = 1e-3, val_split = 0.2, loss = 'categorical_crossentropy', optimizer = 'adam'):
        '''
        Trains the CNN classifier model

        Args:
            model - the CNN model as <tensorflow.keras.models.Model>
            X - Input data as <np.array>
            y - labels as one-hot encoded <np.array>
            model_path - path to save the model as <str>
            epochs - number of epochs to run the training as <int>
            batch_size - batch size for training as <int>
            lr- learning rate for training the model as <float>
            val_split - validation split as <float>
            loss - type of loss function as <str>
            optimizer - type of optimizer as <str>

        Returns:
            history of training as <dict>

        Exception:
            -
        '''
        model.compile(
            loss = loss,
            optimizer = optimizer,
            metrics = ['accuracy']
        )
        history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split = val_split)
        model.save(model_path)
        return history
    
if __name__ == '__main__':
    train_ds = TrafficSign_Dataset()
    X, y = train_ds()
    model = TrafficSign_Model()
    trainer = TrafficSign_Trainer()
    h = trainer.train(model, X, y)