# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt


def training_testing_success_rate_visualization(degrees, success_rate_tr, success_rate_te):
    """visualize the bias variance decomposition."""
    success_rate_tr_mean = np.expand_dims(np.mean(success_rate_tr, axis=0), axis=0)
    
    success_rate_te_mean = np.expand_dims(np.mean(success_rate_te, axis=0), axis=0)
    plt.plot(
        degrees,
        success_rate_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        #label='train',
        linewidth=0.3)
    plt.plot(
        degrees,
        success_rate_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        #label='test',
        linewidth=0.3)
    plt.plot(
        degrees,
        success_rate_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        success_rate_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    plt.ylim(0.65, 0.9)
    plt.xlabel("degree")
    plt.ylabel("fraction correctly predicted")
    plt.legend(loc="best")
    #plt.title("Bias-Variance Decomposition")
    plt.savefig("Images/training_testing_success_rate.pdf")
    

def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    plt.plot(
        degrees,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        label='train',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        label='test',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    plt.ylim(0.2, 0.7)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.title("Bias-Variance Decomposition")
    plt.savefig("bias_variance")