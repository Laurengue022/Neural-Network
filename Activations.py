from Base import Base
from Activation import Activation
import numpy as np


# Hyperbolic tangent activation


class Tanh(Activation):
    def __int__(self):
        def tanh(x):
            return np.tanh(x)

    def tanh_prime(x):
        return 1 - np.tanh(x) ** 2

    super().__int__(tanh, tanh_prime)

    # Solving for the mean squared error > IDK where its suppose to go yet
# def mse(y_true, y_pred):
# return np.mean(np.power(y_true - y_pred, 2))

# def mse_prime(y_true, y_pred):
# return 2 * (y_pred - y_true) / np.size(y_true)
