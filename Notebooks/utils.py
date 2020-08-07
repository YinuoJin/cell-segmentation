import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage as ndi
from sklearn.metrics import accuracy_score, f1_score
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import convex_hull_image


##########################################
#  Plots & Transformation
##########################################


def get_contour(img, external=True, draw=False):
    """ Get contour of image mask or prediction"""
    mode = cv2.RETR_EXTERNAL if external else cv2.RETR_TREE
    img_copy = img.copy()
    img_copy = np.round(img_copy * 255.0).astype(np.uint8)
    _, thresh = cv2.threshold(img_copy, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, mode, cv2.CHAIN_APPROX_NONE)
    
    if not draw:
        return contours
    else:
        img_processed = np.zeros_like(img)
        cv2.drawContours(img_processed, contours, -1, (255, 255, 255), 1)
        img_processed = img_processed / 255.0  # Convert back from [0, 255] to [0, 1]
    
    return contours, img_processed


def hausdorff(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Inconsistent dimensions between prediction and ground-truth matrix"
    hd_list = []
    for r1, r2 in zip(y_true, y_pred):
        r1 = r1.detach().cpu().squeeze().numpy()
        r2 = r2.detach().cpu().squeeze().numpy()
        if y_true.shape[1] == 1:
            hd = max(directed_hausdorff(r1, r2)[0], directed_hausdorff(r2, r1)[0])
        else:
            r1_binary = np.zeros((r1.shape[1], r1.shape[2]))
            r1_binary[np.bitwise_and(r1[1] >= r1[0], r1[1] >= r1[2])] = 1
            r2_binary = np.zeros_like(r1_binary)
            r2_binary[r2[1] == 1] = 1
            hd = max(directed_hausdorff(r1_binary, r2_binary)[0], directed_hausdorff(r2_binary, r1_binary)[0])

        hd_list.append(hd)
        
    return np.mean(hd_list)


def calc_accuracy_score(y_true, y_pred):
    accuracies = []
    for r1, r2 in zip(y_true, y_pred):
        r1 = r1.detach().cpu().numpy()
        r2 = r2.detach().cpu().numpy()
        r1_label, r2_label = np.argmax(r1, axis=0), np.argmax(r2, axis=0)
        accuracies.append(accuracy_score(r1_label.flatten(), r2_label.flatten()))
        
    return np.mean(accuracies)


def calc_f1_score(y_true, y_pred):
    accuracies = []
    for r1, r2 in zip(y_true, y_pred):
        r1 = r1.detach().cpu().numpy()
        r2 = r2.detach().cpu().numpy()
        
        if r1.shape[0] > 1:
            r1_label, r2_label = np.argmax(r1, axis=0), np.argmax(r2, axis=0)
        else:
            r1_label = np.zeros((1, r1.shape[1], r1.shaoe[2]))
            r2_label = np.zeros_like(r1_label)
            r1_label[r1 < 0.5] = 1
            r1_label[1, r1_label[0] == 0] = 1
            r2_label[r2 < 0.5] = 1
            r2_label[1, r2_label[0] == 0] = 1
        accuracies.append(f1_score(r1_label.flatten(), r2_label.flatten(), average='weighted'))
        
    return np.mean(accuracies)


def class_assignment(y_pred, t1, t2):
    """
    Assign class labels from segmentation predictions
    
    Parameters
    ----------
    y_pred  : torch.Tensor
        predicted output matrix (softmax output) shape: (B, C=3, H, W)
    t1 : float
        threshold 1 - cutoff for class 0 & 1 assignment (background & cell foreground)
    t2 : float
        threshold 2 - cutoff for class 2 asssignment (attaching border)
    """
    output = np.zeros_like(y_pred)
    for i, y in enumerate(y_pred):
        curr_output = output[i]
        mask1 = y[2] > t2
        curr_output[mask1] = 2
        mask2 = np.bitwise_and(y[1] > t1, curr_output != 2)
        curr_output[mask2] = 1
        
        # Assign pixels with label=2 to its closest adjacent cells
        c1, c2 = mask1.astype(np.int8), mask2.astype(np.int8)
        dist = ndi.distance_transform_edt(1-c1, return_indices=True) * c2
        dist[dist == 0] = np.inf
        r_coords, c_coords = np.where(dist == dist.min())
        for r, c in zip(r_coords, c_coords):
            curr_output[r, c] = 1
        curr_output[curr_output != 1] = 0
    
    return output


def watershed_indep_masks(g, ft_size=4, thresh=0.1):
    """
    Post-processing prediction matrix - watershed individual concave masks
    
    Parameters
    ----------
    g : np.ndarray
        predicted mask, shape=(h, w)
    ft_size : int
        FIlter size for watershed markers detection
    thresh : float
        threshold value for "Concave" mask determination
        let g_concave = convex(g) \ g, thresh = Area(g_concave) / Area(g)
        
    Returns
    -------
    g_output : np.ndarray
        Output watershed segmented results of each independent predicted mask
    """
    def _g_watershed(indep_mask, ft_size):
        # distance = distance_transform_edt(indep_mask)
        distance = ndi.distance_transform_cdt(indep_mask)
        try:
            local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((ft_size, ft_size)),
                                        labels=indep_mask)
            markers = ndi.label(local_maxi)[0]
            labels = watershed(-distance, markers, mask=indep_mask, watershed_line=True)
            labels_binary = (labels > 0).astype(np.float)
        except IndexError:
            labels_binary = np.zeros_like(indep_mask)
        
        return labels_binary
    
    # Generate n 2D arrays, separating individual masks from each other
    contours = get_contour(g)
    g_output = np.zeros((g.shape[0], g.shape[1]))
    for i in range(len(contours)):
        g_indep_mask = np.zeros_like(g_output)
        contour = contours[i].squeeze(axis=1).astype(np.int32)
        cv2.fillPoly(g_indep_mask, pts=[contour], color=(255, 255, 255))
        g_indep_mask /= 255.0
        g_convex = convex_hull_image(g_indep_mask)
        g_concave_comp = g_convex - g_indep_mask
        
        if (g_concave_comp == 1.0).sum() / (g_indep_mask == 1.0).sum() >= thresh:
            g_ws = _g_watershed(g_indep_mask, ft_size)
            g_output[g_ws == 1] = 1
        else:
            g_output[g_indep_mask == 1] = 1
    
    return g_output
        

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
    # todo: Generalize to multi-class segmentation
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
    # todo: Generalize to multi-class segmentation
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

    
class ShapeBCELoss(nn.Module):
    """
    Shape-aware BCE loss based on distance / shape-based loss functions
    """
    def __init__(self):
        super(ShapeBCELoss, self).__init__()
        
    def forward(self, y_true, y_pred, weight):
        """
        Parameters
        ----------
        y_true : torch.Tensor
            ground truth matrix (one-hot encoded), shape: [B, C, H, W], C = 3
        y_pred : torch.Tensor
            predicted matrix (softmax output), shape: [B, C, H, W], C = 3
        weight : torch.Tensor
            Shape-awared weights of each pixel, highlighting adjacent cell borders, smaller & concave cells
        """
        return F.binary_cross_entropy_with_logits(y_pred, y_true, weight)
