#testing editng
#testing editing 2
import numpy as np

class activation:
    '''
    A class containing various activation functions for neural networks.
    '''

    @staticmethod
    def sigmoid(z):
        '''
        Brief: Compute the sigmoid activation function.

        Argument(s):
        * z (numpy.ndarray) - Input values.

        Returns:
        numpy.ndarray: Output after applying the sigmoid function.
        '''
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def tanh(z):
        '''
        Brief: Compute the hyperbolic tangent (tanh) activation function.

        Argument(s):
        * z (numpy.ndarray) - Input values.

        Returns:
        numpy.ndarray: Output after applying the tanh function.
        '''
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    @staticmethod
    def relu(z):
        '''
        Brief: Compute the Rectified Linear Unit (ReLU) activation function.

        Argument(s):
        * z (numpy.ndarray) - Input values.

        Returns:
        numpy.ndarray: Output after applying the ReLU function.
        '''
        return np.maximum(0, z)

    @staticmethod
    def leaky_relu(z, l=0.01):
        '''
        Brief: Compute the Leaky ReLU activation function.

        Argument(s):
        * z (numpy.ndarray) - Input values.
        * l (float, optional) - Leaky slope for negative values (default is 0.01).

        Returns:
        numpy.ndarray: Output after applying the Leaky ReLU function.
        '''
        return np.maximum(l * z, z)

    @staticmethod
    def prelu(z, l):
        '''
        Brief: Compute the Parameterized ReLU (PReLU) activation function.

        Argument(s):
        * z (numpy.ndarray) - Input values.
        * l (numpy.ndarray) - Learnable parameter for slopes.

        Returns:
        numpy.ndarray: Output after applying the PReLU function.
        '''
        return np.maximum(l * z, z)

    @staticmethod
    def elu(z, l):
        '''
        Brief: Compute the Exponential Linear Unit (ELU) activation function.

        Argument(s):
        * z (numpy.ndarray) - Input values.
        * l (float) - ELU parameter.

        Returns:
        numpy.ndarray: Output after applying the ELU function.
        '''
        return np.maximum(l * (np.exp(z) - 1), z)

    @staticmethod
    def selu(z, l, s):
        '''
        Brief: Compute the Scaled Exponential Linear Unit (SELU) activation function.

        Argument(s):
        * z (numpy.ndarray) - Input values.
        * l (float) - SELU parameter for slopes.
        * s (float) - SELU parameter for scaling.

        Returns:
        numpy.ndarray: Output after applying the SELU function.
        '''
        return s * np.maximum(l * (np.exp(z) - 1), z)

    @staticmethod
    def gelu(z):
        '''
        Brief: Compute the GELU (Gaussian Error Linear Unit) activation function.

        Argument(s):
        * z (numpy.ndarray) - Input values.

        Returns:
            numpy.ndarray: Output after applying the GELU function.
        '''
        arg = np.sqrt(2 / np.pi) * (z + 0.044715 * (z ** 3))
        return (z / 2) * (1 + activation.tanh(arg))

    @staticmethod
    def softmax(z):
        '''
        Brief: Compute the softmax activation function for multi-class classification.

        Argument(s):
        * z (numpy.ndarray) - Input values.

        Returns:
        numpy.ndarray: Output after applying the softmax function.
        '''
        return np.exp(z) / np.sum(np.exp(z))

    @staticmethod
    def swish(z):
        '''
        Brief: Compute the Swish activation function.

        Argument(s):
        * z (numpy.ndarray) - Input values.

        Returns:
        numpy.ndarray: Output after applying the Swish function.
        '''
        return z * activation.sigmoid(z)
