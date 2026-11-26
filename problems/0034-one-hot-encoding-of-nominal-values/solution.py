import numpy as np

def to_categorical(x, n_col=None):
    # Your code here
    if n_col == None:
        encod = np.zeros((len(x), max(x) + 1))
    else:
        encod = np.zeros((len(x), n_col))
    for index, value in enumerate(x):
        encod[index][value] = 1
    return encod 