import numpy as np

def standard_deviation_func(accuracity : list ):

    mean = np.mean(accuracity)
    result = 0
    for x in accuracity:
        result +=  (x - mean) **2
    return result / len(accuracity)
