# -*- coding: utf-8 -*-
import numpy as np
from build_polynomial import build_poly
from model_learning import ridge_regression_least_squares, least_squares
from costs import compute_mse_loss

def build_k_indices(y, k_fold, seed):
    """build k groups of indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree, 
                     compute_weightsFunction, compute_lossFunction):
    """
    selects kth group of indices as test set and rest as training set,
    builds the polynomial features up to degree d
    computes the weights based on the training set with the specified function
    returns losses of training set and testing set with the specified function
    """
    
    #if compute_weightsFunction is None:
    #    print("Taking default least squares stochastic gradient descent")
    #    compute_weightsFunction = lambda y, tx, lambda_: least_squares(
    #        y, tx, gamma=gamma, max_iters=max_iters, batch_size=200)
    
    #if compute_lossFunction is None:
    #    print("Taking default least squares loss function")
    #    compute_lossFunction = compute_mse_loss # without lambda!
        
        
        
    # determine the indices in the training set and those in the test set
    tr_indices = np.concatenate( (k_indices[:k].ravel(), k_indices[k+1:].ravel()) )
    te_indices = k_indices[k]
    
    # select training and testing x and y
    x_tr = x[tr_indices]
    y_tr = y[tr_indices]
    x_te = x[te_indices]
    y_te = y[te_indices]
    
    # build polynomial features
    x_poly_tr = build_poly(x_tr, degree)
    x_poly_te = build_poly(x_te, degree)
    
    
    # find weights using the training data only
    weights_tr = compute_weightsFunction(y_tr, x_poly_tr, lambda_)
    
    # compute the losses for cross validation
    loss_tr = compute_lossFunction(y_tr, x_poly_tr, weights_tr) # compute without lambda
    loss_te = compute_lossFunction(y_te, x_poly_te, weights_tr)
    
    return loss_tr, loss_te

def k_cross_validation(y, x, k_fold, lambda_, degree, seed, 
                       compute_weightsFunction, compute_lossFunction):
    """ do k-fold validation for input data (x,y) and polynomial features up
        to given degree and with regularization constant lambda_
        return the rmse of the mean losses for training and testing
        seed is used to divide data into k groups
        usually, just interested in the testing error
    """
    
    losses_tr = []
    losses_te = []
    
    # construct k groups for cross-validation
    k_indices = build_k_indices(y, k_fold, seed)
        
    # compute training error and testing error for each of k_fold possibilities
    for k in range(k_fold):
        (mse_tr, mse_te) = cross_validation(y, x, k_indices, k, lambda_, degree, 
                                            compute_weightsFunction=compute_weightsFunction, 
                                            compute_lossFunction=compute_lossFunction)
        losses_tr.append(mse_tr)
        losses_te.append(mse_te)
    
    # find validation error of k-fold cross-validation by averaging over the mse
    rmse_tr = np.sqrt(2*np.mean(losses_tr))
    rmse_te = np.sqrt(2*np.mean(losses_te))
    
    return (rmse_tr, rmse_te)