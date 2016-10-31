# implements all the methods to learn the parameters of the model (GD + SGD with logistic and least-squares)
import numpy as np
from costs import *
from helpers import batch_iter
from build_polynomial import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma, lambda_=0, min_loss_threshold = 0):
    """Linear regression using gradient descent """

    print_step = np.maximum(int(max_iters/10), 1) # print status of gradient descent whenever a multiple of this
    
    w = initial_w

    loss_change = min_loss_threshold + 1
    loss = compute_loss_least_squares(y, tx, w, lambda_)
    for n_iter in range(max_iters):
        grad = compute_gradient_least_squares(y, tx, w, lambda_)
        w = w - gamma * grad
        
        old_loss = loss
        loss = compute_loss_least_squares(y, tx, w, lambda_)
        
        loss_change = np.max(np.abs(loss - old_loss))
        if loss_change <= min_loss_threshold:
            break
        #print(w)
        
        if n_iter % print_step == 0:
            print("Gradient Descent({bi}/{ti}): changeInLoss={lc}, loss={l}, w0={w0}, w1={w1}".format(
                lc = loss_change, bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
            
    return (w, loss)



def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1, lambda_=0, min_loss_threshold = 0):
    """ Linear regression using stochastic gradient descent """    
    
    print_step = np.maximum(int(max_iters/10), 1) # print status of gradient descent whenever a multiple of this
    
    w = initial_w

    loss_change = min_loss_threshold + 1
    loss = compute_loss_least_squares(y, tx, w, lambda_)
    n_iter = 0
    while (n_iter < max_iters) and (loss_change > min_loss_threshold):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            if n_iter >= max_iters:
                break
                
            grad = compute_gradient_least_squares(minibatch_y, minibatch_tx, w, lambda_)
            w = w - gamma * grad
            
            old_loss = loss
            loss = compute_loss_least_squares(y, tx, w, lambda_)
            
            loss_change = np.max(np.abs(loss - old_loss))
            if loss_change <= min_loss_threshold:
                break
                
            if n_iter % print_step == 0:
                print("Gradient Descent({bi}/{ti}): changeInLoss={lc}, loss={l}, w0={w0}, w1={w1}".format(
                    lc = loss_change, bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
            
            n_iter = n_iter + 1
    
    return (w, loss)

def least_squares(y, tx):
    """ Least squares regression using normal equations """
    return ridge_regression(y, tx, lambda_ = 0)

def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations """
    
    N, M = tx.shape
    
    # ill-conditioned, for seed = 12, below (Ex2, end), there is a high RMSE error 
    # due to numerics when numbers are very identical (and division)
    # w = np.dot(np.linalg.inv(np.dot(tx.T, tx)), np.dot(tx.T, y))
    w = np.linalg.solve(np.dot(tx.T, tx) + lambda_ * np.identity(M), np.dot(tx.T, y))
    
    loss = compute_loss_least_squares(y, tx, w, lambda_)
    return (w, loss)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent or SGD """
    lamb = 0
    return reg_logistic_regression(y, tx, lamb, initial_w, max_iters, gamma)
   
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, min_loss_threshold = 0):
    """ Regularized logistic regression using GD, actually GD works better here """
    
    print_step = np.maximum(int(max_iters/10), 1) # print status of gradient descent whenever a multiple of this
    
    w = initial_w

    loss_change = min_loss_threshold + 1
    loss = compute_loss_logistic_regression(y, tx, w, lambda_)
    for n_iter in range(max_iters):
        grad = compute_gradient_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * grad
        
        old_loss = loss
        loss = compute_loss_logistic_regression(y, tx, w, lambda_)
        
        loss_change = np.max(np.abs(loss - old_loss))
        if loss_change <= min_loss_threshold:
            break
        #print(w)
        
        if n_iter % print_step == 0:
            print("Gradient Descent({bi}/{ti}): changeInLoss={lc}, loss={l}, w0={w0}, w1={w1}".format(
                lc = loss_change, bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
            
    return (w,loss)

def reg_logistic_regression_with_SGD(y, tx, lambda_, initial_w, max_iters, gamma, min_loss_threshold = 0):
    """ Regularized logistic regression using SGD """

    batch_size = 1
    
    print_step = np.maximum(int(max_iters/10), 1) # print status of gradient descent whenever a multiple of this
    
    w = initial_w

    loss_change = min_loss_threshold + 1
    loss = compute_loss_logistic_regression(y, tx, w, lambda_)
    n_iter = 0
    while (n_iter < max_iters) and (loss_change > min_loss_threshold):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            if n_iter >= max_iters:
                break
                
            grad = compute_gradient_logistic_regression(minibatch_y, minibatch_tx, w, lambda_)
            w = w - gamma * grad
            
            old_loss = loss
            loss = compute_loss_logistic_regression(y, tx, w, lambda_)
            
            loss_change = np.max(np.abs(loss - old_loss))
            if loss_change <= min_loss_threshold:
                break
                
            if n_iter % print_step == 0:
                print("Gradient Descent({bi}/{ti}): changeInLoss={lc}, loss={l}, w0={w0}, w1={w1}".format(
                    lc = loss_change, bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
            
            n_iter = n_iter + 1
    
    return w




# helper functions for gradients and costs



def compute_gradient_least_squares(y, tx, w, lambda_):
    """ with regularization term """
    e = y - tx @ w
    N = tx.shape[0]
    return -1/N*tx.T @ e + 2 * lambda_ * w

def compute_loss_least_squares(y, tx, w, lambda_):
    e = y - tx @ w
    return 1/2*np.mean(e**2) + lambda_ * np.sum(w**2)

def compute_gradient_logistic_regression(y, tx, w, lambda_):
    y_trans = (y == 1) # 0 if y != 1, 1 otherwise
    return tx.T @ ( sigmoid(tx @ w) - y_trans ) + 2 * lambda_ * w
    
def compute_loss_logistic_regression(y, tx, w, lambda_):
    y_trans = (y == 1) # 0 if y != 1, 1 otherwise
    a = np.sum(approximate_ln1pex(tx @ w))
    b = np.sum(y_trans * (tx @ w))
    #print(np.max(a))
    #print(np.max(b))
    return np.sum(approximate_ln1pex(tx @ w)) - np.sum(y_trans * (tx @ w)) + lambda_ * np.sum(w**2)

def approximate_ln1pex(x):
    """ approximates ln(1+e^x) """
    large_x_indices = (x > 30)
    small_x_indices = (x < 30)
    
    res = np.zeros(x.shape)
    res[large_x_indices] = x[large_x_indices]
    res[small_x_indices] = np.log(1 + np.exp(x[small_x_indices]))
    
    return res
    
def sigmoid(x):
    """ approximates sigmoid if it is too large """
    
    positive_x_indices = (x >= 0)
    negative_x_indices = (x < 0)
    
    res = np.zeros(x.shape)

    res[positive_x_indices] = 1 / (1 + np.exp(-x[positive_x_indices]))
    res[negative_x_indices] = np.exp(x[negative_x_indices]) / (1 + np.exp(x[negative_x_indices]))
    
    return res





# helper functions for cross validation


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    p = np.random.permutation(len(x))
    x_tr, x_te = np.split(x[p], [int(ratio*len(x))])
    y_tr, y_te = np.split(y[p], [int(ratio*len(x))])
    return x_tr, y_tr, x_te, y_te

def get_fraction_correct(y, tx, w):
    return np.mean(y == np.sign(tx.dot(w)))


def build_k_indices(y, k_fold, seed):
    """build k groups of indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, tx, k_indices, k, lambda_, degree, cross_features_degree,
                     compute_weights_function, compute_loss_function):
    """
    selects kth group of indices as test set and rest as training set,
    builds the polynomial features up to degree d
    computes the weights based on the training set with the specified function
    returns losses of training set and testing set with the specified function
    """
    

    # determine the indices in the training set and those in the test set
    tr_indices = np.concatenate( (k_indices[:k].ravel(), k_indices[k+1:].ravel()) )
    te_indices = k_indices[k]
    
    # select training and testing x and y
    tx_tr = tx[tr_indices]
    y_tr = y[tr_indices]
    tx_te = tx[te_indices]
    y_te = y[te_indices]
    
    # build polynomial features
    tx_poly_tr = build_poly(tx_tr, degree, cross_features_degree)
    tx_poly_te = build_poly(tx_te, degree, cross_features_degree)
    
    # find weights using the training data only
    nbFeatures = tx_poly_tr.shape[1]
    initial_w = np.array([10] * nbFeatures)
    weights_tr = compute_weights_function(y_tr, tx_poly_tr, lambda_, initial_w)
    
    # compute the losses for cross validation
    loss_tr = compute_loss_function(y_tr, tx_poly_tr, weights_tr, lambda_)
    loss_te = compute_loss_function(y_te, tx_poly_te, weights_tr, lambda_)
    fraction_correct_tr = get_fraction_correct(y_tr, tx_poly_tr, weights_tr)
    fraction_correct_te = get_fraction_correct(y_te, tx_poly_te, weights_tr)
    
    return loss_tr, loss_te, fraction_correct_tr, fraction_correct_te

def k_cross_validation(y, x, degree, cross_features_degree, lambda_, k_fold, seed, 
                       compute_weights_function, compute_loss_function):
    """ do k-fold validation for input data (x,y) and polynomial features up
        to given degree and with regularization constant lambda_
        return the rmse of the mean losses for training and testing
        seed is used to divide data into k groups
        usually, just interested in the testing error
    """
    
    losses_tr = []
    losses_te = []
    fraction_correct_tr = []
    fraction_correct_te = []
    
    # construct k groups for cross-validation
    k_indices = build_k_indices(y, k_fold, seed)
        
    # compute training error and testing error for each of k_fold possibilities
    for k in range(k_fold):
        (mse_tr, mse_te, fraction_tr, fraction_te) = cross_validation(
            y, x, k_indices, k, lambda_, degree, cross_features_degree, 
            compute_weights_function=compute_weights_function, 
            compute_loss_function=compute_loss_function)
        
        losses_tr.append(mse_tr)
        losses_te.append(mse_te)
        fraction_correct_tr.append(fraction_tr)
        fraction_correct_te.append(fraction_te)
    
    # find validation error of k-fold cross-validation by averaging over the mse
    rmse_tr = np.sqrt(2*np.mean(losses_tr))
    rmse_te = np.sqrt(2*np.mean(losses_te))
    
    return (rmse_tr, rmse_te, fraction_correct_tr, fraction_correct_te)
    
    
