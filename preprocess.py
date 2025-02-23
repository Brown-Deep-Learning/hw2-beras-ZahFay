import numpy as np
from beras.onehot import OneHotEncoder
from beras.core import Tensor
from tensorflow.keras import datasets

def load_and_preprocess_data() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    '''This is where we load in and preprocess our data! We load in the data 
        for you but you'll need to flatten the images, normalize the values and 
        convert the input images from numpy arrays into tensors
    Return the preprocessed training and testing data and labels!'''
    
    #Load in the training and testing data from the MNIST dataset
    (train_inputs, train_labels), (test_inputs, test_labels) = datasets.mnist.load_data()

    train_tensor = Tensor(flatten(normalize(train_inputs)))
    train_label_tensor = Tensor(np.array(train_labels))

    test_tensor = Tensor(flatten(normalize(test_inputs)))
    test_label_tensor = Tensor(np.array(test_labels))

    return train_tensor,train_label_tensor,test_tensor,test_label_tensor


def normalize(data):
    return np.array(data / 255.0)

def flatten(data):
    flattened = []
    for value in data:
        flattened.append(value.flatten())
    return np.array(flattened)