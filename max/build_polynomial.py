# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    powers = np.arange(1, degree+1).T
    x_with_poly = x[:, :, None]**powers[None,None,:] # or np.newaxis, use of broadcasting
    # x_with_poly is a 3d array, 3rd dimension indexes the polynomial degree, concat to end
    
    s1, s2, s3 = x_with_poly.shape
    reshaped = x_with_poly.reshape(s1, s2*s3)
    # add ones only here to avoid having number of features for '1' for degree 0
    return np.hstack((np.ones((reshaped.shape[0],1)), reshaped))
    
