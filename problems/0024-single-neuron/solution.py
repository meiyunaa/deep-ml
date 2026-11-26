import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    z = np.array(features) @ np.array(weights) + bias
    probabilities = 1 / (1 + np.exp(-z))
    mse = ((probabilities - labels) ** 2).mean()
    return probabilities, mse