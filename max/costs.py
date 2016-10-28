# implement the loss function and its gradient
import numpy as np



# the loss function of least squares, computes the mse
def compute_mse_loss(y, tx, w):
    """Calculate the mse loss.
    """
    err = y - np.dot(tx, w)
    return 1/2 * np.mean(err**2)

def compute_mae_loss(y, tx, w):
    """Calculate the mae loss.
    """
    err = y - np.dot(tx, w)
    return np.mean(np.abs(err))


# computes the gradient of least squares loss function, is used for gradient descent
def compute_mse_gradient(y, tx, w):
    """Compute the mse gradient."""
    N = len(y)
    err = y - np.dot(tx, w)
    return -1/N*np.dot(tx.T, err)

def compute_mae_gradient(y, tx, w):
    """Compute the mae gradient."""
    #err = np.dot(tx, w) - y
    
    N = len(y)
    
    return 1/N * tx.T.dot(np.sign(tx.dot(w) - y))


def compute_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    
    # problem with large numbers
    #res = - np.sum((y==1) * np.log(sigmoid(tx.dot(w)))) - np.sum((y==-1) * np.log(1-sigmoid(tx.dot(w))))
    #res = - np.sum(np.log(sigmoid(tx[y==1,:].dot(w))))) + np.sum(np.log(sigmoid(tx[y==-1,:].dot(w)))))
    
    #print( np.sum((y==1) * np.log(sigmoid(tx.dot(w)))) )
    #print((y==1).T.dot(np.log(sigmoid(tx.dot(w)))).shape)
    
    #print("1: {}".format(tx.shape))
    #print(((y == 1) * tx.dot(w)).shape)
    #print("1: {}".format( ((y == 1) * tx.dot(w)).shape ))
    #print("2: {}".format( (ln_1_p_exp_x(tx.dot(w))).shape ))
    res = np.sum( ln_1_p_exp_x(tx.dot(w)) - (y == 1) * tx.dot(w) )
    
    return res

# returns ln(1 + exp(x))
def ln_1_p_exp_x(x):
    
    x_gt10 = (x >= 10)
    x_st10 = (x < 10)
    
    res = np.zeros(x.shape)
    res[x_gt10] = x[x_gt10]
    res[x_st10] = np.log(1 + np.exp(x[x_st10]))
    
    return res
    


def compute_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    
    res = tx.T.dot(sigmoid(tx.dot(w)) - y)
    #print(res.T.dot(res))
    return res

def sigmoid(t):
    """apply sigmoid function on t."""
    
    t_gt0_indices = (t >= 0)
    t_st0_indices = (t < 0)
    #t[t < -10] = -10

    res = np.zeros(t.shape)
    res[t_gt0_indices] = 1/(1+np.exp(-t[t_gt0_indices]))
    res[t_st0_indices] = np.exp(t[t_st0_indices])/(1 + np.exp(t[t_st0_indices]))

    #print(res.shape)
    #print(np.where(res == np.nan))
    return res

