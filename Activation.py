import numpy as np

from Base import Base


class Activation(Base):
    # Constructor takes 2 parameters : Activation and its derivative > both are functions

    def __int__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # F Method applies activation to the input

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    # Returns the derivative of the error with respect to the input

    def backward(self, output_gradient, learning_rate):
        # Implementing Vector equality
        return np.multiply(output_gradient, self.activation_prime(self.input))

