import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return Tensor(np.mean(np.mean((y_true - y_pred)**2, axis = 1)))

    def get_input_gradients(self) -> list[Tensor]:
        y_pred = self.inputs[0]
        y_true = self.inputs[1]

        rows = y_pred.shape[0]
        columns = y_pred.shape[1]

        grad_y_pred = (-2/ (rows * columns)) *(y_true - y_pred)
        grad_y_true = (2/ (rows * columns)) *(y_true - y_pred)
        return [grad_y_pred,grad_y_true]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        y_clip = np.clip(y_pred, 1e-10,1.0)
        return - 1 * np.sum(y_true, y_clip)

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred = self.inputs[0]
        y_true = self.inputs[1]
        
        return y_pred - y_true
