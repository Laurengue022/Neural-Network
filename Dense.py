from Base import Base
import numpy as np


class Dense(Base):
    # Constructor > takes in 2 parameters > input size: # of neurons in input, output size: # of neurons in output

    def __int__(self, input_size, output_size):
        # Initialize the weights and biases randomly
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    # F Method computes y = wx + b using numpy dot product function

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    # B Method returns the derivative of the error with respect to the input

    def backward(self, output_gradient, learning_rate):
        # Calculates Derivative of the error with respect to the weights > then, Derivative of the error with respect
        # to the biases
        weights_gradient = np.dot(output_gradient, self.input.T)
        # Updates the parameters with gradient descent
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)
        pass
