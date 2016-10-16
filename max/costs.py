# implement the loss function and its gradient
import numpy as np



# the loss function of least squares, computes the mse
def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    N = len(y)
    err = y - np.dot(tx, w)
    return 1/(2*N)*np.inner(err, err)


# computes the gradient of least squares loss function, is used for gradient descent
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = len(y)
    err = y - np.dot(tx, w)
    return -1/N*np.dot(tx.T, err)