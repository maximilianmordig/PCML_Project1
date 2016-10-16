# implements all the methods to learn the parameters of the linear regression
import numpy as np
from costs import compute_loss, compute_gradient


# 'standard' gradient descent
def gradient_descent(y, tx, initial_w, max_iters, gamma): 
    """Gradient descent algorithm."""
    
    print_step = np.maximum(int(max_iters/20), 1) # print status of gradient descent whenever a multiple of this
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        gradient = compute_gradient(y, tx, w)
        
        w = w - gamma * gradient
        
        loss = compute_loss(y, tx, w)

        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        
        if n_iter % print_step == 0:
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws


# stochastic gradient descent, max_epochs is the maximum number of iterations
# at most max_epochs gradients are computed (independent of batch size)
def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""
    
    print_step = np.maximum(int(max_iters/20), 1) # print status of gradient descent whenever a multiple of this
    
    # Define parameters to store w and loss
    
    ws = [initial_w]
    losses = []
    
    w = initial_w
    n_iter = 0
    while n_iter < max_epochs:
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size,shuffle=True):
            if n_iter >= max_epochs:
                break
                
            gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma*gradient
            
            loss = compute_loss(y, tx, w)
            # store
            ws.append(np.copy(w))
            losses.append(loss)
        
            if n_iter % print_step == 0:
                print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
            n_iter = n_iter + 1
    
    return losses, ws


# computes the least squares solution with the normal equations
def least_squares(y, tx):
    """calculate the least squares solution."""
    
    # weights
    # ill-conditioned, for seed = 12, below (Ex2, end), there is a high RMSE error 
    # due to numerics when numbers are very identical (and division)
    # w = np.dot(np.linalg.inv(np.dot(tx.T, tx)), np.dot(tx.T, y))
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    
    loss = compute_loss(y, tx, w)
    
    return (loss, w)


# compute ridge regression solution using normal equations
def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    
    N, M = tx.shape
    lambModif = lamb#*(2*N)
    
    w = np.linalg.solve(np.dot(tx.T, tx) + lambModif * np.identity(M), np.dot(tx.T, y))
    # mean square error
    err = y - np.dot(tx, w)
    
    loss = compute_loss(y, tx, w) # without lambda
    #mse = 1/(2*N) * np.dot(err.T, err)# + lamb*np.dot(w.T,w)
    
    return (loss, w)