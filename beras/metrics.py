import numpy as np

from beras.core import Callable


class CategoricalAccuracy(Callable):
    #from handout: probs represents the probability of each class as predicted by the model
    #labels is a one hot encoded vector representing the true model class.
    def forward(self, probs, labels):
        ## HINT: Argmax + boolean mask via '=='
        arg_probs = np.argmax(probs, axis = 1)
        arg_labels = np.argmax(labels, axis = 1)
        return np.mean(arg_probs == arg_labels)
