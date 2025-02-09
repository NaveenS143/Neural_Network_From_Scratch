from layer import Layer
from activation import Activation
import numpy as np

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)
        
        def tanh_derivative(x):
            return 1 - np.tanh(x) ** 2
        
        super().__init__(tanh, tanh_derivative)