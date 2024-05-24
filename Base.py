class Base:
    def __int__(self):
        self.input = None  # Attribute 1 > does NOT need to be declared by other layers
        self.output = None  # Attribute 2 > does NOT need to be declared by other layers

    # Methods

    # Takes in the input and gives you the output
    def forward(self, input):
        # Return output
        pass

    # Takes in the derivative of the error with respect to the output (output_gradient) Responsible for 2 things:
    # 1.Updating the trainable parameters 2.Returning the derivative of the error with respect to the input of the layer

    def backward(self, output_gradient, learning_rate):
        # Update parameters and return input gradient
        pass
