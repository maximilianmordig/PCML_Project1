# implements all the methods to learn the parameters of the linear regression
import numpy as np
from costs import *
from helpers import batch_iter

# 'standard' gradient descent
def gradient_descent(y, tx, initial_w, max_iters, gamma, compute_gradientFunction, compute_lossFunction): 
    """Gradient descent algorithm."""
    
    N = tx.shape[0]
    
    # takes more time because shuffling (even if won't change result for GD) takes time
    return stochastic_gradient_descent(
        y, tx, initial_w, batch_size=N, max_iters=max_iters, gamma=gamma, 
        compute_gradientFunction=compute_gradientFunction, compute_lossFunction=compute_lossFunction)


# stochastic gradient descent, max_epochs is the maximum number of iterations
# at most max_iters iterations (calls a sample of size 'batch_size' up to 'max_iters' times)
def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma, compute_gradientFunction, compute_lossFunction):
    """Stochastic gradient descent algorithm."""
    
    #if compute_gradientFunction is None:
    #    print("Taking default least squares gradient")
    #    compute_gradientFunction = compute_mse_gradient
    
    #if compute_lossFunction is None:
    #    print("Taking default least squares loss function")
    #    compute_lossFunction = compute_mse_loss # without lambda!
    
        
    N = y.shape[0]
    isSGD = (batch_size < N)
    SGD_string = "Stochastic " if isSGD else ""
        
    print_step = np.maximum(int(max_iters/10), 1) # print status of gradient descent whenever a multiple of this
    
    # Define parameters to store w and loss
    
    #ws = [initial_w]
    #losses = []
    
    w = initial_w
    n_iter = 0
    min_change_threshold = 10e-4
    last_change = 1 # in loss function
    loss = compute_lossFunction(y, tx, w)
    while last_change > min_change_threshold and n_iter < max_iters:
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size,shuffle=True):
            if n_iter >= max_iters or last_change < min_change_threshold:
                break
                
            gradient = compute_gradientFunction(minibatch_y, minibatch_tx, w)
            w = w - gamma*gradient
            
            old_loss = loss
            loss = compute_lossFunction(y, tx, w)
            # store
            #ws.append(np.copy(w))
            #losses.append(loss)
        
            last_change = abs(loss - old_loss)
            if n_iter % print_step == 0:
                print(SGD_string + "Gradient Descent({bi}/{ti}): changeInLoss={lc}, loss={l}, w0={w0}, w1={w1}".format(
                      lc = last_change, bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
            n_iter = n_iter + 1
            
    
    return w
    #return losses, ws

    
# computes the least squares solution with the normal equations
def exact_least_squares_weights(y, tx):
    """calculate the least squares solution."""
    
    return exact_ridge_regression_least_squares(y, tx, lamb=0)


# computes the least squares solution plus lambda with the normal equations
def exact_ridge_regression_least_squares(y, tx, lamb):
    """calculate the least squares solution with regularization lambda/2N."""
    
    N, M = tx.shape
    lambModif = lamb#*(2*N)
    
    # ill-conditioned, for seed = 12, below (Ex2, end), there is a high RMSE error 
    # due to numerics when numbers are very identical (and division)
    # w = np.dot(np.linalg.inv(np.dot(tx.T, tx)), np.dot(tx.T, y))
    w = np.linalg.solve(np.dot(tx.T, tx) + lambModif * np.identity(M), np.dot(tx.T, y))
    return w


    
# computes the least squares solution of least squares using SGD (stochastic gradient descent)
def least_squares_weights(y, tx, gamma, max_iters, batch_size=None):
    """calculate the least squares solution."""
    
    return ridge_regression_least_squares(y, tx, lamb=0, gamma=gamma, 
                                          max_iters=max_iters, batch_size=batch_size)


# compute ridge regression solution of least squares using SGD (stochastic gradient descent)
def ridge_regression_least_squares_weights(y, tx, lamb, gamma, max_iters, batch_size=None):
    """implement ridge regression for least squares."""
    
    return ridge_regression_method(y, tx, lamb, gamma, max_iters, compute_gradientFunction=compute_mse_gradient, 
                            compute_lossFunction=compute_mse_loss, batch_size=batch_size)

# analogous for mae
def mae_weights(y, tx, gamma, max_iters, batch_size=None):
    """calculate the weights of the mae solution."""
    
    return ridge_regression_mae_weights(y, tx, lamb=0, gamma=gamma, 
                                          max_iters=max_iters, batch_size=batch_size)

# analogous for mae
def ridge_regression_mae_weights(y, tx, lamb, gamma, max_iters, batch_size=None):
    
    return ridge_regression_method(y, tx, lamb, gamma, max_iters, compute_gradientFunction=compute_mae_gradient, 
                            compute_lossFunction=compute_mae_loss, batch_size=batch_size)


# computes the least squares solution of least squares using SGD (stochastic gradient descent)
def logistic_weights(y, tx, gamma, max_iters, batch_size=None):
    """calculate the least squares solution."""
    
    return ridge_regression_logistic_weights(y, tx, lamb=0, gamma=gamma, 
                                          max_iters=max_iters, batch_size=batch_size)


def ridge_regression_logistic_weights(y, tx, lamb, gamma, max_iters, batch_size=None):
    
    return ridge_regression_method(y, tx, lamb, gamma, max_iters, compute_gradientFunction=compute_logistic_gradient, 
                            compute_lossFunction=compute_logistic_loss, batch_size=batch_size)



# general ridge regression, adds lambda to gradient (leaves unchanged the loss function)
def ridge_regression_method(y, tx, lamb, gamma, max_iters, compute_gradientFunction, 
                            compute_lossFunction, batch_size=None):
    # compute_gradientFunction is without lambda! 
    
    N, M = tx.shape
    
    if batch_size is None:
        batch_size = N
        
    # TODO: lambda/2N?
    compute_ridge_gradient = lambda y1, tx1, w: compute_gradientFunction(y1, tx1, w) + 2 * lamb * w
    compute_ridge_lossFunction = lambda y1, tx1, w: compute_lossFunction(y1, tx1, w) + lamb * sum(w**2)
    
    initial_w = np.array([10]*M)
    
    w = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, 
                                    gamma, compute_gradientFunction=compute_ridge_gradient,
                                    compute_lossFunction=compute_ridge_lossFunction)
    
    return w

# implements logistic regression
