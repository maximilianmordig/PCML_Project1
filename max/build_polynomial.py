# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np
import operator as op
import functools
    
# global variables to avoid duplication in cross-features
maskIndicesDict = {}

def computeMaskForCrossFeatureDegree(cross_features_degree, nbFeatures):
    """
    changes the global variables 'maskIndices' and 'computedForCrossDegree'
    to avoid recomputing mask each time
    """
    
    global maskIndicesDict
    
    key = (cross_features_degree, nbFeatures)
    if not (key in maskIndicesDict):
        print("Recomputing for cross_features_degree {}, num_features {}".format(cross_features_degree, nbFeatures))
        
        maskPrimes = generate_mask(nbFeatures, cross_features_degree)
        uniq, maskIndices = np.unique(maskPrimes, return_index=True)
        
        maskIndicesDict[key] = maskIndices
        

def build_poly(x, degree, cross_features_degree): 
    """polynomial basis functions for input data x, for j=0 up to j=degree.
        builds correlated features like xy, just 1, x, x², y, y²
    """
    
    global maskIndicesDict
    
    nbFeatures = x.shape[1]
    key = (cross_features_degree, nbFeatures)
    if not (key in maskIndicesDict):
        # recomputeMaskForCrossFeatureDegree not called
        raise Exception('Mask must be computed for this degree {} before calling this function'.format(cross_features_degree))
    
    maskIndices = maskIndicesDict[key]
    
    # assumes mask corresponds to this cross_features_degree, always contains constant term (even for degree 0)
    poly_order1_crossterms = build_poly_and_crossterms(x, cross_features_degree, maskIndices)
        
    # build non-cross related features for higher degrees
    powers = np.arange(cross_features_degree+1, degree+1).T
    x_with_poly = x[:, :, None]**powers[None,None,:] # or np.newaxis, use of broadcasting
    # x_with_poly is a 3d array, 3rd dimension indexes the polynomial degree, concat to end
    
    s1, s2, s3 = x_with_poly.shape
    x_with_poly = x_with_poly.reshape(s1, s2*s3)
    
    # We don't add ones, they were already added with the crossterms (always contains constant term, also for degree 0
    return np.hstack((poly_order1_crossterms, x_with_poly))




# builds crossterms, e.g. if [x, y] is input and degree 3,
# it builds all terms 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3,
# avoids duplicates (xy=yx) by setting mask generated
# with generate_mask
def build_poly_and_crossterms(x, degree, maskIndices=None):
    num_entries_before1, num_features_before1 = x.shape
    x = np.hstack((np.ones((num_entries_before1, 1)), x))
    num_entries, num_features = x.shape
  
    if maskIndices is None:
        final = np.zeros((num_entries, num_features**degree))
    else:
        final = np.zeros((num_entries, ncr(degree+num_features-1, degree)))

    # builds degree+1 dimensional matrix and then puts all together in the end
    # only selects unique indices if mask is given (that indexes the unique indices)
    for n in range(num_entries):
        res = np.array([1])
        for i in range(1, degree+1):
            temp = x[n, :].reshape([num_features] + [1]*i)
            res = res[np.newaxis,:] * temp
    
        if maskIndices is None:
            final[n, :] = res.ravel()
        else:
            final[n, :] = res.ravel()[maskIndices]

    return final

# implements 'n choose r'
def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = functools.reduce(op.mul, range(n, n-r, -1))
    denom = functools.reduce(op.mul, range(1, r+1))
    return numer//denom

# check if number is prime
def is_prime(a):
    return all(a % i for i in range(2, a))

# get 'num_primes' primes starting at 2
def prime_range(num_primes):
    prime_numbers = [2]
    i = 3
    while len(prime_numbers) < num_primes:
        if is_prime(i):
            prime_numbers.append(i)
        i+=2

    return prime_numbers

# generate mask, use to find unique features to make x*y and y*x identical
def generate_mask(num_features, degree):
    prime_numbers = prime_range(num_features)
    return build_poly_and_crossterms(np.array([prime_numbers]), degree)