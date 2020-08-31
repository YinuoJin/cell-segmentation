import cv2
import torch
import numpy as np
import pandas as pd

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries
from skimage.morphology import convex_hull_image
from utils import get_contour


class Postprocessor():
    """Postprocess predicted matrix with 2 rounds of watershed segmentation"""
    def __init__(self, x, y_pred, seg_mask=True, return_binary=True, return_contour=True):
        """
        Parameters
        ----------
        x : torch.Tensor
            input image shape:(B=1, C=1, H, W)
        y_pred  : torch.Tensor
            predicted output matrix (softmax output) shape: (B=1, C=3, H, W)
        seg_mask : bool
            whether to apply "mask" regions for seeded watershed segmentation
        return_binary : bool
            return binarized segmentation prediction, if False each individual masks will be assigned with distinct integer
        return_contour : bool
            return contour of the segmentation prediction
        """
        self.x = x.detach().cpu().squeeze().numpy()
        self.y_pred = y_pred.detach().cpu().squeeze().numpy()
        self.return_binary = return_binary
        self.return_contour = return_contour
        if seg_mask:
            self.seg_region = np.bitwise_or(self.y_pred[1] > 0.5, self.y_pred[2] > 0.5)
        else:
            self.seg_region = None
        assert self.x.ndim == 2, 'Invalid dimension of input image {0}'.format(self.x.shape)
        assert self.y_pred.ndim == 3, 'Invalid dimension of predicted matrix {0}, only accepts batchsize=1'.format(self.y_pred.shape)

    @property
    def out(self):
        mask_pred_binary = self.class_assignment(0.5, 0.5)  # binarize 3-channel prediction
        
        # 2 rounds of watershed segmentation
        mask_pred_shape = self.watershed_shape(mask=mask_pred_binary, ft_size=8, thresh=0.1)
        mask_pred_seed = self.watershed_seed(img=self.x,
                                             mask=mask_pred_shape,
                                             seg_region=self.seg_region,
                                             return_binary=self.return_binary)
        self._out = find_boundaries(mask_pred_seed) if self.return_contour else mask_pred_seed
        
        return self._out.astype(np.float)
        
    def class_assignment(self, t1, t2):
        """
        Assign class labels from segmentation predictions

        Parameters
        ----------
        t1 : float
            threshold 1 - cutoff for class 0 & 1 assignment (background & cell foreground)
        t2 : float
            threshold 2 - cutoff for class 2 asssignment (attaching border)
        """
        n_channels, h, w = self.y_pred.shape
        output = np.zeros((h, w))
        mask1 = self.y_pred[2] > t2
        output[mask1] = 2
        mask2 = np.bitwise_and(self.y_pred[1] > t1, output != 2)
        output[mask2] = 1
        output[output == 2] = 0
        
        # Assign pixels with label=2 to its closest adjacent cells
        c1, c2 = mask1.astype(np.int8), mask2.astype(np.int8)
        dist = ndi.distance_transform_edt(1 - c1, return_indices=False) * c2
        dist[dist == 0] = np.inf
        output[np.where(dist == dist.min())] = 1
        output[output != 1] = 0
        
        return output
    
    @staticmethod
    def watershed_shape(mask, ft_size=4, thresh=0.1):
        """
        Post-processing the prediction matrix - watershed individual concave masks
    
        Parameters
        ----------
        mask : np.ndarray
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
        def _watershed_indep_mask(indep_mask, ft_size):
            """Perform watershed segmentation on independent mask"""
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
        contours = get_contour(mask)
        mask_output = np.zeros((mask.shape[0], mask.shape[1]))
        for i in range(len(contours)):
            indep_mask = np.zeros_like(mask_output)
            contour = contours[i].squeeze(axis=1).astype(np.int32)
            cv2.fillPoly(indep_mask, pts=[contour], color=(255, 255, 255))
            indep_mask /= 255.0
            mask_convex = convex_hull_image(indep_mask)
            mask_concave_comp = mask_convex - indep_mask
            
            if (mask_concave_comp == 1.0).sum() / (indep_mask == 1.0).sum() >= thresh:
                mask_ws = _watershed_indep_mask(indep_mask, ft_size)
                mask_output[mask_ws == 1] = 1
            else:
                mask_output[indep_mask == 1] = 1
        
        return mask_output

    @staticmethod
    def watershed_seed(img, mask, seg_region=None, min_perim=2, sigma=-1, return_binary=True):
        """
        Post-processing the prediction matrix - seed-watershed
        
        Parameters
        ----------
        mask : np.ndarray
             predicted mask, shape=(h, w)
        seg_region : np.ndarray
            candidate regions for watershed segmentation, shape=(h, w)
        min_perim : int
            minimum perimeter of the independent mask to be considered as a candidate for seeded watershed
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
        mask_coords = mask_coords_raw[mask_coords_raw.apply(lambda x: (x == True).sum() >= min_perim)]
        
        raw_seeds = mask_coords.apply(lambda x: np.array(np.where(x == True)).mean(1).astype(np.int16))
        seeds = np.stack(raw_seeds.to_numpy())
        marker_bools = np.zeros_like(img)
        marker_bools[tuple(seeds.T)] = 1
        markers = ndi.label(marker_bools)[0]
        
        # second-round watershed based on the centroids of the first-round watershed
        labels = watershed(img, markers=markers, mask=seg_region, watershed_line=True)
        labels_binary = labels > 0
        
        return labels_binary if return_binary else labels
