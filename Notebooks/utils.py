import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D

##########################################
#  Plots & Transformation
##########################################

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
    
    
def get_contour(img):
    """Get contour of image mask (ground truth or prediction)"""
    img = img.detach().squeeze().numpy()
    img = np.round(img * 255.0).astype(np.uint8) # Convert from [0, 1] to [0, 255]
    _, thresh = cv2.threshold(img, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_processed = np.zeros_like(img)
    cv2.drawContours(img_processed, contours, -1, (255, 255, 255), 1)
    img_processed = img_processed / 255.0 # Convert back from [0, 255] to [0, 1]
    
    return torch.Tensor(np.expand_dims(img_processed, axis=0))


##########################################
#  Loss functions
##########################################


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) / Jaccard loss:
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    
    reference: https://github.com/LIVIAETS/surface-loss/blob/master/losses.py
    """
    
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, y_true, y_pred):
        """
        Parameters
        ----------
        y_true : torch.Tensor
            ground truth matrix, shape: [B, C, H, W], C = 1
        y_pred : torch.Tensor
            predicted matrix, shape: [B, C, H, W], C = 1
        """
        intersection = torch.einsum('bchw,bchw->bc', y_true, y_pred)
        total = torch.einsum('bchw->bc', y_true) + torch.einsum('bchw->bc', y_pred)
        union = total - intersection
        
        iou = ((intersection + self.smooth) / (union + self.smooth)).mean()
        
        return 1 - iou


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss:
    Dice(A, B) = (2 * |A ∩ B|) / (|A| + |B|)
    
    reference: https://github.com/LIVIAETS/surface-loss/blob/master/losses.py
    """
    
    def __init__(self, smooth=1e-6):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, y_true, y_pred):
        """
        Parameters
        ----------
        y_true : torch.Tensor
            ground truth matrix, shape: [B, C, H, W], C = 1
        y_pred : torch.Tensor
            predicted matrix, shape: [B, C, H, W], C = 1
        """
        intersection = torch.einsum('bchw,bchw->bc', y_true, y_pred)
        total = torch.einsum('bchw->bc', y_true) + torch.einsum('bchw->bc', y_pred)
        
        dice = ((2.0 * intersection + self.smooth) / (total + self.smooth)).mean()
        
        return 1 - dice
    
    
class SurfaceLoss(nn.Module):
    """
    Combination of Region based loss & Contour (boundary)-based loss
    
    reference: https://github.com/LIVIAETS/surface-loss/blob/master/losses.py
    """
    
    def __init__(self, alpha=0.5, dice=True):
        """
        Parameters
        ----------
        alpha : float
            rate of Region-based loss w.r.t to Boundary-based loss: L = α * L_r + (1 - α) * L_b
        dice : bool
            whether to use Soft Dice loss as region-based loss (use Jaccard loss if False)
        """
        super(SurfaceLoss, self).__init__()
        self.alpha = alpha
        self.use_dice = dice
        self.region_loss = SoftDiceLoss() if dice else IoULoss()
        
    def forward(self, y_true, y_pred, theta_true):
        """
        Parameters
        ----------
        y_true : torch.Tensor
            ground truth matrix, shape: [B, C, H, W], C = 1
        y_pred : torch.Tensor
            predicted matrix, shape: [B, C, H, W], C = 1
        theta_true : torch.Tensor
            level set representation of ground-truth boundary, shape: [B, C, H, W], C = 1
        """
        boundary_dist = torch.einsum('bchw,bchw->bchw', y_pred, theta_true)
        boundary_loss = boundary_dist.mean()
        region_loss = self.region_loss(y_true, y_pred)
        
        return self.alpha * region_loss + (1 - self.alpha) * boundary_loss


class DistWeightBCELoss(nn.Module):
    """
    Unet-like weighted BCE loss (adding distmaps of each pixel to its closest two foreground neighbors
    """

    def __init__(self):
        super(DistWeightBCELoss, self).__init__()

    def forward(self, y_true, y_pred, weight):
        """
        Parameters
        ----------
        y_true : torch.Tensor
            ground truth matrix, shape: [B, C, H, W], C = 1
        y_pred : torch.Tensor
            predicted matrix, shape: [B, C, H, W], C = 1
        weight : torch.Tensor
            Combined weights of distance & unbalanced class labels for BCE loss calculation, shape: [B, C, H, W], C = 1
        """
        return F.binary_cross_entropy_with_logits(y_pred, y_true, weight)
