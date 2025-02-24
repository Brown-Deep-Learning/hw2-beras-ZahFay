import numpy as np

from .core import Diffable,Tensor

class Activation(Diffable):
    @property
    def weights(self): return []

    def get_weight_gradients(self): return []


################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Activation):

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.x = None

    def forward(self, x) -> Tensor:
        """Leaky ReLu forward propagation!"""
        self.x = np.array(x)
        #Same as the logic of, if x > 0 then keep x as is, otherwise replace it with self.alpha * x
        #np.where is used due to the fact that we have an np.array to iterate over
        return Tensor(np.where(self.x > 0, self.x, self.alpha * self.x))

    def get_input_gradients(self) -> list[Tensor]:
        """
        Leaky ReLu backpropagation!
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        #Wherever the input is greater than 0, switch it 1, otherwise switch it to alpha
        return [Tensor(np.where(np.array(self.inputs) > 0, 1, self.alpha))]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J

class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self):
        super().__init__(alpha=0)


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Activation):
    
    def forward(self, x) -> Tensor:
        npx = np.array(x)
        #np.exp(x) = e^x
        return Tensor(1/(1 + np.exp(-npx)))

    def get_input_gradients(self) -> list[Tensor]:
        """
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        npx = np.array(self.inputs)
        sig_x = 1/(1 + np.exp(-npx))
        return [Tensor(np.array(sig_x*(1-sig_x)))]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J


class Softmax(Activation):
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    ## TODO [2470]: Implement for default output activation to bind output to 0-1

    def forward(self, x):
        """Softmax forward propagation!"""
        ## Not stable version
        ## exps = np.exp(inputs)
        ## outs = exps / np.sum(exps, axis=-1, keepdims=True)
        npx = np.array(x)
        ## HINT: Use stable softmax, which subtracts maximum from
        ## all entries to prevent overflow/underflow issues
        max_entry = np.exp(npx - np.max(npx,axis = -1, keepdims = True))
        return max_entry/np.sum(max_entry, axis = -1, keepdims = True)

    def get_input_gradients(self):
        """Softmax input gradients!"""
        x, y = self.inputs + self.outputs
        bn, n = x.shape
        grad = np.zeros(shape=(bn, n, n), dtype=x.dtype)

        #calculate softmax output, fill diagonal of jacobian matrix with i=j case
        #then fill everything else with -SS which is similar to the outer product of matrix
        for i in range(bn):
            soft_out = y[i]
            np.fill_diagonal(grad[i], soft_out*(1- soft_out))
            grad[i] = grad[i] - np.outer(soft_out,soft_out)
        raise grad