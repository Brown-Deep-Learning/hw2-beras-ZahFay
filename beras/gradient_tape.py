from collections import defaultdict

import numpy as np

from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        To find what methods are available on certain objects, reference the cheat sheet
        """

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs.

        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]} 
        counter = 0
        while len(queue) != 0:
            counter+= 1
            current_tensor = queue.pop()
            current_id = id(current_tensor)
            layer = self.previous_layers[current_id] #gives the list of layers previous to the current layer, since it's linear we expect it to be a single one
            if layer != None:
                inputs = layer.inputs
                print("running compose input")
                input_grad = layer.compose_input_gradients(grads[current_id])
                # print("length of inputs, " ,len(inputs))
                print("finshed compose input")
                # print("length of inputs grad, " ,len(input_grad))
                for inp, grad in zip(inputs,input_grad):
                    grads[id(inp)] = grad
                    queue.append(inp)
                print("running compose weight")
                weights = layer.weights
                weight_grad = layer.compose_weight_gradients(grads[current_id])
                print("finshed compose weight")
                # print("length of weight, " ,len(weights))
                # print("length of weight grad, " ,len(weight_grad))
                for wei, grad in zip(weights,weight_grad):
                    grads[id(wei)] = grad
                    queue.append(wei)

        output = []
        for source in sources:
            output.append(grads[id(source)][0])
        return output
        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful
