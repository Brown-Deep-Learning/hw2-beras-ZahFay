from types import SimpleNamespace
from beras.activations import ReLU, LeakyReLU, Softmax
from beras.layers import Dense
from beras.losses import CategoricalCrossEntropy, MeanSquaredError
from beras.metrics import CategoricalAccuracy
from beras.onehot import OneHotEncoder
from beras.optimizers import Adam
from preprocess import load_and_preprocess_data
import numpy as np

from beras.model import SequentialModel

#do batch_size 512, epochs = 10
def get_model():
    model = SequentialModel(
        [
           # Add in your layers here as elements of the list!
           # e.g. Dense(10, 10),
           Dense(784,256, "kaiming"),
           LeakyReLU(),
           Dense(256,128,"kaiming"),
           Softmax()
        ]
    )
    return model

def get_optimizer():
    # choose an optimizer, initialize it and return it!
    adam_opt = Adam(learning_rate=0.005)
    return adam_opt

def get_loss_fn():
    # choose a loss function, initialize it and return it!
    cross = CategoricalCrossEntropy()
    return cross

def get_acc_fn():
    # choose an accuracy metric, initialize it and return it!
    acc = CategoricalAccuracy()
    return acc

if __name__ == '__main__':

    ### Use this area to test your implementation!

    # 1. Create a SequentialModel using get_model

    # 2. Compile the model with optimizer, loss function, and accuracy metric
    
    # 3. Load and preprocess the data
    
    # 4. Train the model

    # 5. Evaluate the model
    my_model = get_model()
    my_model.compile(get_optimizer, get_loss_fn, get_acc_fn)
    data = load_and_preprocess_data()
    my_model.fit(data[0], data[1], 10, 32)
    my_model.fit(data[2], data[3], 32)