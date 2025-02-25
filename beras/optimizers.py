from collections import defaultdict
import numpy as np

class BasicOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def apply_gradients(self, trainable_params, grads):
        trainable_params.assign(trainable_params - grads*self.learning_rate)


class RMSProp:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = defaultdict(lambda: 0)

    def apply_gradients(self, trainable_params, grads):
        for i in range(len(trainable_params)):
            param = trainable_params[i]
            grad = grads[i]

            # paramID = id(param)
            # print(paramID)
            #populate dictionary
            self.v[i] = self.beta * self.v[i] + ((1- self.beta) * np.square(grad))

            #Calculate new parameters
            denominator = np.sqrt(self.v[i]) + self.epsilon
            trainable_params[i].assign(param - (self.learning_rate / denominator) * grad) 
            # param.assign(param - (self.learning_rate / denominator) * grad)



class Adam:
    def __init__(
        self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    ):


        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = defaultdict(lambda: 0)         # First moment zero vector
        self.v = defaultdict(lambda: 0)         # Second moment zero vector.
        self.t = 0                              # Time counter

    def apply_gradients(self, trainable_params, grads):
        return NotImplementedError
