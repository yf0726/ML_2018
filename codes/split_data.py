# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np
def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    idx = list(range(0,len(x)))
    np.random.seed(seed)
    np.random.shuffle(idx)
    training_idx = idx[:int(ratio*len(x))]
    test_idx = idx[int(ratio*len(x)):]
    [x_train,y_train,x_test,y_test] = [x[training_idx], y[training_idx], x[test_idx], y[test_idx]]
    return x_train,y_train,x_test,y_test
    # ***************************************************