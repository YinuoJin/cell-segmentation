import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D


# +---------------------+
# |       plots         |
# +---------------------+

def plot_img_3d_distribution(img, figsize=(8, 6)):
    """
    Plot 3D value distribution of the givne image,
    assume image has shape [C, H, W]"""
    img = img.transpose((1, 2, 0))  # change image order to [H, W, C]
    height, width = img.shape[0], img.shape[1]
    if img.shape[-1] == 1:
        img = img.squeeze()
    else:
        img = img.mean(-1)
    
    img_vals = np.zeros((height * width, 3))
    img_vals[:, 0] = np.repeat(np.arange(height), width)
    img_vals[:, 1] = np.tile(np.arange(width), height)
    img_vals[:, 2] = img.flatten()
    
    df = pd.DataFrame(img_vals, columns=['X', 'Y', 'Z'])
    
    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
    ax.view_init(45, 45)
    plt.show()
    plt.close()


def plot_img_histogram(img, figsize=(8, 6)):
    """Image histogram visualization, assume image has shape [C, H, W]"""
    # reference: https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
    img = img.transpose((1, 2, 0))  # change image order to [H, W, C]
    hist, bins = np.histogram(img.flatten(), 256, [0, 1])
    
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    
    plt.figure(figsize=figsize)
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 1])
    plt.xlim([0, 1])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()
    plt.close()


# +---------------------+
# |   loss functions    |
# +---------------------+


def vgg_topo_loss(model, y_true, y_pred, weight=1, mu=0.1):
    """
    Calculate topo-aware loss function (Mosinska et al. 2018) utilizing feature maps of VGG19
    
    Parameters
    ----------
    model : torch.nn
        Pretrained VGG19 model
    y_true : torch.Tensor
        Ground truth matrix, shape: [batch_size, C, H, W]
    y_pred : torch.Tensor
        Prediction matrix, shape: [batch_size, C, H, W]
    weight : float
        weight for BCE loss calculation
    mu : float
        multiplier of topo loss
        
    Returns
    -------
    loss : float
        Combined loss value of bce loss and topo loss
    """
    
    def _calc_dist(arr):
        idx = len(arr) // 2
        return np.power(distance.euclidean(arr[:idx], arr[idx:]), 2)
    
    bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=torch.Tensor([weight]))

    y_true = torch.cat((y_true, y_true, y_true), axis=1)
    y_pred = torch.cat((y_pred, y_pred, y_pred), axis=1)
    features_true = model(y_true)
    features_pred = model(y_pred)
    topo_loss = 0
    
    for f_pred, f_true in zip(features_pred, features_true):
        vec1 = f_pred.detach().squeeze().numpy()
        vec2 = f_true.detach().squeeze().numpy()
        assert vec1.shape == vec2.shape
        
        n_channels, height, width = vec1.shape
        multiplier = 1 / (n_channels * height * width)
        vec = np.concatenate([vec1.reshape(n_channels, -1), vec2.reshape(n_channels, -1)], axis=1)
        dist = np.sum(
            np.apply_along_axis(_calc_dist, axis=1, arr=vec)
        )
        
        topo_loss += multiplier * dist
        
    return bce_loss + mu * topo_loss
