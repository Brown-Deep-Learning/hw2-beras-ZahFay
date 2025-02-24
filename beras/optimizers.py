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

    ##TODO: Figure out how to get rid of discrepency 
    def apply_gradients(self, trainable_params, grads):
       print(trainable_params)
       
       #update dictionary
       self.v[trainable_params] = self.beta * self.v[trainable_params]


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
    
    #  for parameter, grad in zip(trainable_params, grads):
    #         #parameter as a tensor is not hashable, so we get its ID to make it the key for our dictionary
    #         param_ID = id(parameter)
            
    #         #update dictionary
    #         self.v[param_ID] = self.beta * self.v[param_ID] + ((1- self.beta) * np.square(grad))

    #         #calculations for trainable params
    #         denom = (np.sqrt(self.v[param_ID]))+ self.epsilon
    #         partPara = (self.learning_rate/denom) * grad

    #         #update the parameter
    #         parameter.assign(parameter - partPara)
