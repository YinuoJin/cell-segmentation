import cv2
import torch
import numpy as np
import pandas as pd

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries
from skimage.morphology import convex_hull_image, dilation, opening, closing, square, disk
from utils import get_contour, class_assignment


class Postprocessor():
    """Postprocess predictions with shape-based watershed segmentation"""
    def __init__(self, pred, p=0.9, return_binary=True, return_contour=True):
        """
        Parameters
        ----------
        pred : torch.Tensor
            predicted output matrix (softmax output) shape: (B=1, C=3, H, W)
        p : float
            cutoff ratio of concave complementary area to convex hull area
        return_binary : bool
            return binarized segmentation prediction
        return_contour : bool
            return contour of the segmentation prediction
        """
        self.pred = pred.detach().cpu().squeeze(0).numpy()
        self.return_binary = return_binary
        self.return_contour = return_contour
        self.p = p
        assert self.pred.ndim == 3, "Invalid dimension of predicted matrix {0}, only accepts batchsize=1".format(self.pred.shape)

    @property
    def out(self):
        if self.pred.shape[0] == 3:
            mask_pred_binary = class_assignment(self.pred, 0.5, 0.5)
            mask_pred_binary, mask_labels = label_masks(mask_pred_binary, return_contour=False)
            seg_region = np.bitwise_or(mask_pred_binary > 0.5, self.pred[2] > 0.5)
        else:  # binary predictions
            mask_pred_binary = self._binarize(self.pred.squeeze())
            mask_pred_binary, mask_labels = label_masks(mask_pred_binary, return_contour=False)
            seg_region = mask_pred_binary

        mask_output = watershed_shape(mask_pred_binary, mask_labels, thresh=self.p)
        mask_final = watershed_seed(img=-ndi.distance_transform_edt(mask_pred_binary),
                                    mask=mask_output,
                                    seg_region=seg_region)
        mask_final = opening(mask_final, square(3))

        return mask_final


    def _binarize(self, mask, thresh=0.5):
        return (mask > thresh).astype(np.float)


def label_masks(mask, return_contour=False):
    """Separate and label individual masks, remove internal holes"""
    contours = get_contour(mask)
    mask_indep_labels = np.zeros_like(mask)
    for i in range(len(contours)):
        indep_mask = np.zeros_like(mask)
        contour = contours[i].squeeze(axis=1).astype(np.int32)
        cv2.fillPoly(indep_mask, pts=[contour], color=(255, 255, 255))
        indep_mask /= 255.0

        # remove internal holes in individual mask
        indep_ext_contour = get_contour(closing(indep_mask, square(2)))[0].squeeze(axis=1).astype(np.int32)
        cv2.fillPoly(indep_mask, pts=[indep_ext_contour], color=(255, 255, 255))
        indep_mask /= 255.0
        mask_indep_labels[indep_mask != 0] = i + 1

    mask_processed = (mask_indep_labels != 0).astype(np.float)

    if return_contour:
        return mask_processed, mask_indep_labels, contours
    else:
        return mask_processed, mask_indep_labels


def watershed_shape(mask, mask_labels, thresh=0.1):
    """
    Post-processing the prediction matrix - watershed individual concave masks

    Parameters
    ----------
    mask : np.ndarray
        predicted mask, shape=(h, w)
    ft_size : int
        FIlter size for watershed markers detection
    thresh : float
        threshold for "Convex-like" mask via solidity
        solidity = Area(g) / Area(convex(g))
    
    Returns
    -------
    g_output : np.ndarray
        Output watershed segmented results of each independent predicted mask
    """

    def watershed_indep_mask(indep_mask, ft_size):
        """Perform watershed segmentation on independent mask"""
        distance = ndi.distance_transform_edt(indep_mask)
        try:
            local_maxi = peak_local_max(distance, min_distance=ft_size, indices=False,
                                        labels=indep_mask)
            markers = ndi.label(local_maxi)[0]
            labels = watershed(-distance, markers=markers, mask=indep_mask, watershed_line=True)
            labels_binary = (labels > 0).astype(np.float)
        except IndexError:
            labels_binary = np.zeros_like(indep_mask)

        return labels_binary

    mask_output = np.zeros_like(mask)
    unique_label = np.unique(mask_labels)[1:]

    convex_labels, concave_labels = [], []
    diameters = []
    for i in unique_label:
        indep_mask = (mask_labels == i).astype(np.float)
        mask_convex = convex_hull_image(indep_mask).astype(np.float)
        solidity = (indep_mask).sum() / (mask_convex).sum()
        if solidity > thresh:
            dm = np.sqrt(indep_mask.sum() / np.pi) * 2  # "diameter" of "convex-like" masks
            diameters.append(dm)
            mask_output[indep_mask == 1] = 1
            convex_labels.append(i)
        else:
            concave_labels.append(i)

    avg_mask_area = np.vectorize(lambda x: (mask_labels == x).sum())(np.array(convex_labels)).mean()
    ft_size = int(np.round(np.mean(diameters)))
    for label in concave_labels:
        indep_mask = (mask_labels == label).astype(np.float)
        if (indep_mask).sum() > avg_mask_area:
            mask_ws = watershed_indep_mask(indep_mask, ft_size=ft_size)
            mask_output[mask_ws == 1] = 1
        else:
            mask_output[indep_mask == 1] = 1

    return mask_output


def watershed_seed(img, mask, seg_region=None, min_area=5, sigma=-1, return_binary=True):
    """
    Post-processing the prediction matrix - seed-watershed

    Parameters
    ----------
    mask : np.ndarray
        predicted mask, shape=(h, w)
    seg_region : np.ndarray
        candidate regions for watershed segmentation, shape=(h, w)
    min_area : int
        minimum area of any independent mask to be considered for seeded watershed
    sigma : float
        parameterfor gaussian blur
    return_binary : bool
        whether to return binarized prediction mask

    Returns
    -------
    prediction matrix after the seeded watershed , shape=(h, w)
    """
    if sigma > 0:  # Perform gaussian blur on input image
        img = ndi.gaussian_filter(img, sigma=sigma)

    # Find centroids of each individual first-round watershed mask
    mask_labels = ndi.label(mask)[0]
    unique_mask_labels = pd.Series(np.unique(mask_labels)[1:])
    mask_coords_raw = unique_mask_labels.apply(lambda label: np.vectorize(lambda mask: mask == label)(mask_labels))

    # Filter out tiny contours (noises)
    mask_coords = mask_coords_raw[mask_coords_raw.apply(lambda x: (x == True).sum() > min_area)]

    raw_seeds = mask_coords.apply(lambda x: np.array(np.where(x == True)).mean(1).astype(np.int16))
    seeds = np.stack(raw_seeds.to_numpy())
    marker_bools = np.zeros_like(img)
    marker_bools[tuple(seeds.T)] = 1
    marker_bools = dilation(marker_bools, disk(2))
    markers = ndi.label(marker_bools)[0]

    # second-round watershed: markers selected as the centroids of the first-round watershed
    labels = watershed(img, markers=markers, mask=seg_region, watershed_line=True)
    labels_binary = (labels > 0).astype('float')

    return labels_binary if return_binary else labels