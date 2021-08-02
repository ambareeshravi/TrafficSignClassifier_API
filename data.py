'''
Author: Ambareesh Ravi
Date: Jul 31, 2021
Title: data.py
Description:
    Contains the data handling functionalities
'''

from utils import *
from tensorflow.keras.utils import to_categorical

class TrafficSign_Dataset:
    # Creates the Traffic sign dataset
    def __init__(self, data_path = "TrafficSign_Images/", image_size = (32, 32), isTrain = True, isNormalized = True, isDerived = False):
        '''
        Initializes the class

        Args:
            data_path - Parent path to the dataset as <str>
            image_size - size of the inputs as W,H as <tuple>
            isTrain - If it is training model as <bool>
            isNormalized - if the data is to be normalized as <bool>
            isDerived - To use like an abstract class as <bool>
            
        Returns:
            -
        
        Exception:
            -
        '''
        self.image_size = image_size
        self.isTrain = isTrain
        self.path_tag = "train" if self.isTrain else "test"
        self.data_path = os.path.join(data_path, self.path_tag)
        self.isNormalized = isNormalized
        self.test_csv = os.path.join(data_path, "Test.csv")
        signs_df = pd.read_csv(os.path.join(data_path, "sign_names.csv"))
        self.idx_class = dict([(s_id, name) for s_id, name in zip(np.array(signs_df['ClassId']), np.array(signs_df['SignName']))])
        
        self.isDerived = isDerived
        if not self.isDerived:
            self.data, self.labels = self.create_train_data() if self.isTrain else self.create_test_data()
        
    def resize_im_array(self, im):
        '''
        Resizes the image array

        Args:
            im - image array as <nparray>
            
        Returns:
            resized image array as <np.array>
        
        Exception:
            -
        '''
        return cv2.resize(im, self.image_size)
    
    def read_image(self, path):
        '''
        Reads image into the memory

        Args:
            path - image path as <str>
            
        Returns:
            image array as <np.array>
        
        Exception:
            -
        '''
        im = cv2.imread(path)
        im = self.resize_im_array(im)
        return im
    
    def create_train_data(self,):
        '''
        Creates and stores the train data into memory

        Args:
            -
            
        Returns:
            data as <np.array>, labels as <np.array>
        
        Exception:
            -
        '''
        data, labels = list(), list()
        for d in tqdm(read_directory_content(self.data_path)):
            class_name = os.path.split(d)[-1]
            for fl in read_directory_content(d):
                try:
                    im_array = self.read_image(fl)
                    data.append(im_array)
                    labels.append(int(class_name))
                except Exception as e:
                    print(e)
                    pass
        data, labels = np.array(data).astype(np.float32), np.array(labels).astype(np.uint8)
        if self.isNormalized: data /= 255.
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        return data[indices], to_categorical(labels[indices], len(self.idx_class))
    
    def create_test_data(self,):
        '''
        Creates and stores the test data into memory

        Args:
            -
            
        Returns:
            data as <np.array>, labels as <np.array>
        
        Exception:
            -
        '''
        data, labels = list(), list()
        test_df = pd.read_csv(self.test_csv)
        files = np.array(test_df['Path'])
        labels = np.array(test_df['ClassId'])
        
        for fl in tqdm(files):
            im_array = self.read_image(os.path.join(self.data_path, fl.replace("Test/", "")))
            data.append(im_array)
        data = np.array(data).astype(np.float32)
        if self.isNormalized: data /= 255.
        return data, labels
    
    def __call__(self):
        '''
        Returns (data, labels) when called

        Args:
            -
            
        Returns:
            (data as <np.array>, labels as <np.array>)
        
        Exception:
            -
        '''
        return self.data, self.labels