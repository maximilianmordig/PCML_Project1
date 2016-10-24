# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
from costs import *
import csv


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1, max_rows=(50 if sub_sample else None))
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1, max_rows=(50 if sub_sample else None))
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    #if sub_sample:
    #    yb = yb[::50]
    #    input_data = input_data[::50]
    #    ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

            

# loads data from csv file
#def load_data(filename, sub_sample=True):
#    """Load data and convert it to the metrics system."""
#    path_dataset = filename
#    
#    feature_data = np.genfromtxt(filename, delimiter=",", skip_header=1, max_rows=(50 if sub_sample else None))
#    
#    # important to set dtype=None, otherwise, it will become an array of tuples!
#    predictionToNumber = lambda x: -1 if b'b' in x else 1
#    prediction_id_data = np.genfromtxt(filename, delimiter=",", skip_header=1, max_rows=(50 if sub_sample else None), 
#                  usecols=[0,1], converters={0:int, 1:predictionToNumber}, dtype=None)

#    x = feature_data[:, 2:]
#    ids = prediction_id_data[:, 0]
#    predictions = prediction_id_data[:, 1]
    
    
#    return ids, predictions, x


# normalizes the input features x to have zero mean and unit variance
# and appends 1 to features to allow for offset parameter w0
def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    # don't add ones, will get appended in build_poly
    #tx = np.hstack((np.ones((x.shape[0],1)), x))
    tx = x
    return tx, mean_x, std_x



def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]






