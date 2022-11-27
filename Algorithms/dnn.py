"""
Neural Network Implementation
### Authors
- B811082 Baek Jun Young / jupiterbjy@gmail.com
"""

import pickle
from typing import Tuple, List

import numpy as np
from loguru import logger

from samples import n_pairwise
from commons import rmse, numerical_gradient


class VariableLayeredNetwork:
    def __init__(self, input_size, output_size, *hidden_sizes, init_weight=0.01):
        """
        Args:
            input_size:
            *hidden_sizes:
            output_size:
            init_weight:
        """
        # TODO: fill docs

        # Initialize weights, accounting for arbitrary num. of hidden layers
        self.weight = [
            init_weight * np.random.randn(size_1, size_2)
            for size_1, size_2 in n_pairwise(
                (input_size, *hidden_sizes, output_size)
            )
        ]

        # initialize biases, accounting for arbitrary num. of hidden layers
        self.bias = [
            np.zeros(size) for size in (*hidden_sizes, output_size)
        ]

    def forward(self, x):
        """Predict output from input x.
        Args:
            x:
        Returns:
        """

        # TODO: fill docs
        # TODO: if not working, try using ReLU in hidden layer and feed sample as bits
        # TODO: switch to backward propagation

        a = x
        for weight, bias in zip(self.weight[:-1], self.bias[:-1]):
            # using identity func on hidden layer
            a = np.dot(a, weight) + bias

        # using identity func on output layer
        y = np.dot(a, self.weight[-1]) + self.bias[-1]

        return y

    def loss(self, x, t) -> float:
        """Loss function doubling as accuracy. Uses RMSE.
        References:
            https://inistory.tistory.com/111
            https://stats.stackexchange.com/a/194992
        Args:
            x: estimated output
            t: expected outcome
        Returns:
            Error
        """
        # logger.debug("Calculating")

        return rmse(self.forward(x), t)

    def accuracy(self, x, t):
        return self.loss(x, t)

    def gradiant_descent(self, x, t) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Returns Gradiant for weight and bias respectively.
        Args:
            x:
            t:
        Returns:
        """

        def loss(_):
            return self.loss(x, t)

        logger.debug("Calculating weight gradiant")
        grad_weight = [numerical_gradient(loss, w) for w in self.weight]

        logger.debug("Calculating bias gradiant")
        grad_bias = [numerical_gradient(loss, b) for b in self.bias]

        return grad_weight, grad_bias

    def dump_param(self):
        """Dumps parameters"""

        data = (self.weight, self.bias)
        return pickle.dumps(data)

    def load_param(self, b_str: bytes):
        """Loads parameters"""

        self.weight, self.bias = pickle.loads(b_str)