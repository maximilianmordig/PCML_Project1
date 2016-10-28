# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

maskToCompute=[]# maskToCompute is used as a global variable
recompute = True

def build_poly(x, degree): 
    """polynomial basis functions for input data x, for j=0 up to j=degree.
        does not build correlated features like xy, just 1, x, x², y, y²
    """
    
    cross_features_degree = 4
    global maskToCompute
    global recompute
    if len(maskToCompute) == 0 or recompute == True:
        print("Recomputed")
        recompute = False
        num_features = x.shape[1]
        mask = generate_mask(num_features, cross_features_degree)
        uniq, mask = np.unique(mask, return_index=True)
        maskToCompute.append(mask)
    mask = maskToCompute[0]
    tes = np.arange(1,40)
    #print("bHere: {}".format(mask))
    
    poly_order1_crossterms = build_poly_and_crossterms(x, cross_features_degree, mask)
    
    powers = np.arange(cross_features_degree+1, degree+1).T
    x_with_poly = x[:, :, None]**powers[None,None,:] # or np.newaxis, use of broadcasting
    # x_with_poly is a 3d array, 3rd dimension indexes the polynomial degree, concat to end
    
    s1, s2, s3 = x_with_poly.shape
    x_with_poly = x_with_poly.reshape(s1, s2*s3)
    # add ones only here to avoid having number of features for '1' for degree 0
    #return np.hstack((np.ones((reshaped.shape[0],1)), reshaped))
    
    print(poly_order1_crossterms.shape)
    print(x_with_poly.shape)
    return np.hstack((poly_order1_crossterms, x_with_poly))




    
def build_poly_and_crossterms(x, degree, mask=None):
    num_entries, num_features = x.shape
    x = np.hstack((np.ones((num_entries, 1)), x))
    num_entries, num_features = x.shape
  
    if mask is None:
        final = np.zeros((num_entries, num_features**degree))
    else:
        final = np.zeros((num_entries, ncr(degree+num_features-1, degree)))

    for n in range(num_entries):
        res = np.array([1])
        for i in range(1, degree+1):
            temp = x[n, :].reshape([num_features] + [1]*i)
            res = res[np.newaxis,:] * temp
    
        if mask is None:
            final[n, :] = res.ravel()
        else:
            final[n, :] = res.ravel()[mask]

    return final

import operator as op
import functools
def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = functools.reduce(op.mul, range(n, n-r, -1))
    denom = functools.reduce(op.mul, range(1, r+1))
    return numer//denom

def is_prime(a):
    return all(a % i for i in range(2, a))

def prime_range(num_primes):
    prime_numbers = [2]
    i = 3
    while len(prime_numbers) < num_primes:
        if is_prime(i):
            prime_numbers.append(i)
        i+=2

    return prime_numbers

def generate_mask(num_features, degree):
    mask = prime_range(num_features)
    return build_poly_and_crossterms(np.array([mask]), degree)