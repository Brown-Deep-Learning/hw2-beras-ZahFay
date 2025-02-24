import numpy as np
import random

from typing import Literal
from beras.core import Diffable, Variable, Tensor

DENSE_INITIALIZERS = Literal["zero", "normal", "xavier", "kaiming", "xavier uniform", "kaiming uniform"]

class Dense(Diffable):

    def __init__(self, input_size, output_size, initializer: DENSE_INITIALIZERS = "normal"):
        self.w, self.b = self._initialize_weight(initializer, input_size, output_size)
        self.output_size = output_size
        self.x = None

    @property
    def weights(self) -> list[Tensor]:
        return [self.w, self.b]

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for a dense layer! Refer to lecture slides for how this is computed.
        """
        self.x = x
        return (x @ self.w) + self.b

    def get_input_gradients(self) -> list[Tensor]:
        tensor_grad = Tensor(self.w)
        return [tensor_grad]

    def get_weight_gradients(self) -> list[Tensor]:
        #bias array of ones
        array_bias = np.ones(np.array(self.inputs).shape[0])
        array_weight = np.expand_dims(self.inputs[0], axis = -1)
        return [Tensor(array_weight),Tensor(array_bias)]
    
    @staticmethod
    def generateDistribution(input_size,output_size,type):
        weights = np.empty((input_size,output_size))
        for i in range(input_size):
            for j in range(output_size):
                if type == 1: #normal
                    weights[i,j] = random.gauss(0.0,1.0)
                elif type == 2: #xavier-GlorotNormal
                    stand_dev = np.sqrt(2.0/ (input_size + output_size))
                    weights[i,j] = random.gauss(0.0,stand_dev)
                elif type == 3: #kaiming-HeNormal
                    stand_dev = np.sqrt(2.0/ input_size)
                    weights[i,j] = random.gauss(0.0,stand_dev)
        return weights
    

    @staticmethod
    def _initialize_weight(initializer, input_size, output_size) -> tuple[Variable, Variable]:
        """
        Initializes the values of the weights and biases. The bias weights should always start at zero.
        However, the weights should follow the given distribution defined by the initializer parameter
        (zero, normal, xavier, or kaiming). You can do this with an if statement
        cycling through each option!

        Details on each weight initialization option:
            - Zero: Weights and biases contain only 0's. Generally a bad idea since the gradient update
            will be the same for each weight so all weights will have the same values.
            - Normal: Weights are initialized according to a normal distribution.
            - Xavier: Goal is to initialize the weights so that the variance of the activations are the
            same across every layer. This helps to prevent exploding or vanishing gradients. Typically
            works better for layers with tanh or sigmoid activation.
            - Kaiming: Similar purpose as Xavier initialization. Typically works better for layers
            with ReLU activation.
        """

        initializer = initializer.lower()
        assert initializer in (
            "zero",
            "normal",
            "xavier",
            "kaiming",
        ), f"Unknown dense weight initialization strategy '{initializer}' requested"

        #set weights and biases based on the type of distribution
        weight_tensor = None
        bias_tensor = None
        if initializer == "zero":
            weight_tensor = Variable(Tensor(np.zeros((input_size,output_size))))
            bias_tensor = Variable(Tensor(np.zeros((input_size,output_size))))
        elif initializer == "normal":
            weight_tensor = Variable(Tensor(Dense.generateDistribution(input_size, output_size, 1)))
            bias_tensor = Variable(Tensor(np.zeros((output_size,))))
        elif initializer == "xavier":
            weight_tensor = Variable(Tensor(Dense.generateDistribution(input_size, output_size, 2)))
            bias_tensor = Variable(Tensor(np.zeros((output_size,))))
        elif initializer == "kaiming":
            weight_tensor = Variable(Tensor(Dense.generateDistribution(input_size, output_size, 3)))
            bias_tensor = Variable(Tensor(np.zeros((output_size,))))

        return weight_tensor, bias_tensor



