from collections import namedtuple
import logging

import matplotlib.pyplot as plt
import numpy as np


# A data class to group activation functions and their derivatives
ActivationFunction = namedtuple("ActivationFunction", ["name", "function", "derivative"])

sig_func = lambda x: 1 / (1 + np.exp(-x))
sigmoid = ActivationFunction(
    name="Sigmoid",
    function=sig_func,
    derivative=lambda x: sig_func(x) * (1 - sig_func(x))
)

# https://www.wolframalpha.com/input?i=derivative+%281-e%5E-x%29+%2F+%281%2Be%5E-x%29
hyper_tan = ActivationFunction(
    name="Hyperbolic Tangent",
    function=lambda x: (1 - np.exp(-x)) / (1 + np.exp(-x)),
    derivative=lambda x: (2 * np.exp(x)) / ((np.exp(x) + 1) ** 2)
)

# Global list to be used in GUI
ACTIVATION_FUNCTIONS = {
    "Sigmoid": sigmoid,
    "Hyperbolic Tangent": hyper_tan,
}

def get_logger(name):
    fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
    date_fmt = "%H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=date_fmt,
        filename='run.log',
        filemode='w'
    )
    return logging.getLogger(name)

def plot_mses(mses, name):
    plt.title(name + " trainig")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.plot(range(len(mses)), mses)

    plt.show()

class ConfusionMatrix:

    def __init__(self, dimension):
        self.dimension = dimension
        self.matrix = [[0 for _ in range(dimension)] for _ in range(dimension)]

    def add(self, actual, predicted):
        self.matrix[actual][predicted] += 1

    def __repr__(self):
        slots = len(str(max(map(max, self.matrix))))
        aloc = lambda x: str(x) + (" " * (slots - len(str(x))))
        result = ""
        result += "  " + " ".join(map(aloc, range(self.dimension)))

        for i, vals in enumerate(self.matrix):
            result += "\n" + str(i) + " " + " ".join(map(aloc, vals))

        return result
