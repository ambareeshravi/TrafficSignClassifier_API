'''
Author: Ambareesh Ravi
Date: Jul 31, 2021
Title: prediction.py
Description:
    Contains the tester class for the Traffic Sign classifer model
'''

from train import *
from tensorflow.keras.models import load_model

class Tester(TrafficSign_Dataset):
    # Contains functionalities for testing, predicting and evaluating the Traffic Sign Classifier model

    def __init__(self, model_path = "models/Sign_model.h5"):
        '''
        Initializes the class

        Args:
            model_path - path to the trained model as <str>
            
        Returns:
            -
        
        Exception:
            -
        '''
        # Load the model from path
        self.model = load_model(model_path)
        # Derive the dataset class to inherit some data params and member functions
        TrafficSign_Dataset.__init__(self, isDerived = True)
        
    def evaluate(self, X_test, y_test):
        '''
        Evaluates the model on test data

        Args:
            X_test - test data as <np.array>
            y_test - test labels as <np.array>
            
        Returns:
            -
        
        Exception:
            -
        '''
        loss, accuracy = self.model.evaluate(X_test, to_categorical(y_test, y_test.max()+1))
        print("Test Loss: %0.4f | Test Accuracy: %0.4f"%(loss, accuracy))
        
    def predict(self, X):
        '''
        Predicts on test data

        Args:
            X - test data as <np.array>
            
        Returns:
            list of class names
        
        Exception:
            -
        '''
        pred = self.model(X)
        y_pred = np.argmax(pred, axis = -1)
        return [self.idx_class[yp] for yp in y_pred]
    
    def predict_im_array(self, im_array):
        '''
        Predicts on test image array

        Args:
            im_array - test data as <np.array> of shape W, H, C
            
        Returns:
            list of class names
        
        Exception:
            -
        '''
        X = np.expand_dims(im_array, axis = 0)
        return self.predict(X)
        
    def predict_image(self, path):
        '''
        Predicts on test image

        Args:
            path - path to the image as <str>
            
        Returns:
            list of class names
        
        Exception:
            -
        '''
        im_array = self.read_image(path)
        X = np.expand_dims(im_array, axis = 0)
        return self.predict(X)[-1]
        
if __name__ == '__main__':
    # Define tester
    tester = Tester()
    # Load test data
    ds = TrafficSign_Dataset(isTrain = False)
    X, y = ds()
    # Evaluate results
    tester.evaluate(X, y)