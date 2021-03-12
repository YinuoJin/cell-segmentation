import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy.spatial.distance import directed_hausdorff, cosine
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


def recover_3d(file_name):
    """
    Recover 3d np.array from saved txt file

    input dim: (C, HxW) --> output dim: (C, H, W) (assume H = W)
    """
    img = np.loadtxt(file_name).astype(np.float)
    n_layers, npos = img.shape
    img = img.reshape(n_layers, int(np.sqrt(npos)), -1)

    return img


def class_assignment(mask, t1=0.5, t2=0.5):
    """
    Assign class labels from multi-label segmentation predictions

    Parameters
    ----------
    t1 : float
        threshold 1 - cutoff for class 0 & 1 assignment
        (background & cell foreground)

    t2 : float
        threshold 2 - cutoff for class 2 asssignment
        (attaching border)
    """
    n_channels, h, w = mask.shape
    output = np.zeros((h, w))
    mask1 = mask[2] > t2
    output[mask1] = 2
    mask2 = np.bitwise_and(mask[1] > t1, output != 2)
    output[mask2] = 1

    # Assign pixels with label=2 to its closest adjacent cells
    c1, c2 = mask1.astype(np.int8), mask2.astype(np.int8)
    dist = ndi.distance_transform_edt(1 - c1, return_indices=False) * c2
    dist[dist == 0] = np.inf
    output[np.where(dist == dist.min())] = 1
    output[output != 1] = 0

    return output


def hausdorff(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Inconsistent dimensions between prediction and ground-truth matrix"
    hd_list = []
    for r1, r2 in zip(y_true, y_pred):
        r1 = r1.detach().cpu().squeeze().numpy()
        r2 = r2.detach().cpu().squeeze().numpy()
        if y_true.shape[1] == 1:  # 1-channel predictions
            hd = max(directed_hausdorff(r1, r2)[0], directed_hausdorff(r2, r1)[0])
        else:  # 3-channel predictions: convert to binary maps and calculate hausdorff distance
            r1_binary = class_assignment(r1)
            r2_binary = class_assignment(r2)
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
            r1_label = class_assignment(r1)
            r2_label = class_assignment(r2)
        accuracies.append(f1_score(r1_label.flatten(), r2_label.flatten(), average='weighted'))

    return np.mean(accuracies)


def calc_mse_score(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    mse_score = np.einsum('bhw -> b', np.power(y_true - y_pred, 2)).mean()

    return mse_score


"""
def calc_fpn_score(y_true, y_pred):
    # 
    # Calculate hybrid accuracy score from multi-task FPN model
    # 
    # debug: try inner & outer distance only
    # assert y_pred.ndim == 4 and y_pred.shape[1] == 3, "Invalid shape of prediction matrix {0}".format(y_pred.shape)
    # inner_pred, outer_pred, class_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2].unsqueeze(1)
    # inner_gt, outer_gt, class_gt = y_true[:, 0], y_true[:, 1], y_true[:, 2].unsqueeze(1)
    inner_pred, outer_pred = y_pred[:,0], y_pred[:,1]
    inner_gt, outer_gt = y_true[:,0], y_true[:,1]

    inner_acc = calc_mse_score(inner_gt, inner_pred)
    outer_acc = calc_mse_score(outer_gt, outer_pred)
    # class_acc = torch.mean(((class_gt > 0.5) == (class_pred > 0.5)).float()).detach().cpu().numpy()

    return np.mean([inner_acc, outer_acc])
"""


def plot_img_3d_distribution(img, figsize=(8, 6)):
    """
    Plot 3D value distribution of the given image,
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
            rate of Region-based loss w.r.t to Boundary-based loss:
            L = α * L_r + (1 - α) * L_b

        dice : bool
            whether to use Soft Dice loss as region-based loss
            (use Jaccard loss if False)
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
            ground truth matrix
            shape: [B, C, H, W], C = 1

        y_pred : torch.Tensor
            predicted matrix
            shape: [B, C, H, W], C = 1

        theta_true : torch.Tensor
            level set representation of ground-truth boundary
            shape: [B, C, H, W], C = 1
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
            ground truth matrix (one-hot encoded)
            shape: [B, C, H, W], C = 3

        y_pred : torch.Tensor
            predicted matrix (softmax output)
            shape: [B, C, H, W], C = 3

        weight : torch.Tensor
            Shape-awared weights of each pixel
            highlighting adjacent cell borders, smaller & concave cells
        """
        return F.binary_cross_entropy_with_logits(y_pred, y_true, weight)


# todo: decide whether still necessary to keep "direction net + watershed net" option (end-to-end U-net architecture)
"""
class DirectNetMSAELoss(nn.Module):
    #
    #Mean-square-angular error loss (a.k.a "Direction loss") for DN-Unet training
    #
    def __init__(self):
        super(DirectNetMSAELoss, self).__init__()

    def forward(self, y_true, y_pred, weight, eps=1e-10):
        #
        Parameters
        ----------
        y_true : torch.Tensor
            ground truth direction unit-vector matrix, shape: [B, C, H, W], C = 2 (x & y decomposition of the vector)
        y_pred : torch.Tensor
            predicted direction unit-vector matrixm shape: [B, C, H, W], C = 2
        weight : torch,Tensor
            Shape & area awared weights of each pixel, ignore background region and weight higher at cell boundary & smaller cells
        #
        cos_dist = 1 - F.cosine_similarity(y_true, y_pred) # b, h, w
        cos_loss = torch.einsum('bchw, bhw -> b', weight, cos_dist).mean()

        return cos_loss
"""

"""
class FPNLoss(nn.Module):
    # 
    # Loss functions for Multi-task FPN model
    # 
    def __init__(self, alpha=0.1):
        super(FPNLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_true, y_pred, weight):
        #
        #Parameters
        #---------
        #y_true : torch.Tensor
        #    ground truth FPN matrices, shape: [B, C, H, W], C = 3 --> (1). Inner distance; (2). Outer distance; (3). Pixel-wise gt mask
        #y_pred : torch.TEnsor
        #    predicted FPN matrices, shape: [B, C, H, W], C = 3
        #weight : torch.Tensor
        #    Shape & area awared weights of each pixel
        #
        # debug: try only inner and outer distance

        #inner_gt, outer_gt, class_gt = y_true[:,0], y_true[:,1], y_true[:,2]
        #inner_pred, outer_pred, class_pred = y_pred[:,0], y_pred[:,1], y_pred[:,2]
        outer_gt, class_gt = y_true[:,0], y_true[:,1]
        outer_pred, class_pred = y_pred[:,0], y_pred[:,1]

        # MSE loss for inner & outer predictions, Weighted BCELoss for pixel-wise predictions
        outer_loss = F.mse_loss(outer_pred, outer_gt)
        # class_loss = F.binary_cross_entropy_with_logits(class_pred, class_gt, weight.squeeze(1))
        class_loss = F.mse_loss(class_pred, class_gt)

        # loss = (inner_loss + outer_loss) + self.alpha * class_loss
        # loss = outer_loss + self.alpha * class_loss
        loss = outer_loss + class_loss

        return loss
"""
