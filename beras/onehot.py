import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        np_data = np.array(data)
        #first get the number of possible labels
        labels = np.unique(np_data)
        #dictionary for label - onehot vector pair
        match_lv = {}

        for i in range(len(labels)):
            vector = np.zeros(len(labels))
            vector[i] = 1
            match_lv[labels[i]] = vector

    def forward(self, data):
        np_data = np.array(data)
        labels = np.unique(np_data)

        #dictionary for label - index pair
        label_to_index = {}
        for i in range(len(labels)):
            label_to_index[labels[i]] = i

        #create 2d array of one-hot encoded vectors
        vector_options = np.zeros((len(np_data),len(labels)))
        for row in range(len(np_data)):
            label_index = label_to_index[np_data[row]]
            vector_options[row,label_index] = 1
        
        return vector_options

    def inverse(self, data):
        #data here is the vector_options from above, we find all the unique one-hot vectors
        np.unique(data, axis= 0)
        return NotImplementedError
