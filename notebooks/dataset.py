import os
import shutil
import numpy as np
import pandas as pd
import cv2
import torch
import time
import Augmentor
import multiprocessing
import gc

from progress.bar import ChargingBar
from torch.utils import data
from scipy.stats import mode
from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt
from skimage.exposure import adjust_gamma, is_low_contrast, match_histograms, rescale_intensity
from skimage.morphology import black_tophat, convex_hull_image, skeletonize, square, disk, dilation, erosion,reconstruction
from skimage.filters import threshold_mean, threshold_local
from skimage.transform import resize
from skimage.segmentation import find_boundaries

from utils import get_contour
from model import Unet


class ImageDataLoader(data.Dataset):
    """Load raw images and masks"""
    
    def __init__(self, mat_frame, mat_mask):
        super(ImageDataLoader, self).__init__()
        self.mat_frame = mat_frame
        self.mat_mask = mat_mask
        
        # check data dimension
        assert self.mat_frame.shape[0] == self.mat_mask.shape[0], \
            "Inconsistent number of images between frames & masks"
        assert self.mat_frame.shape[2] == self.mat_mask.shape[2] and \
               self.mat_frame.shape[3] == self.mat_mask.shape[3], \
            "Inconsistent image dimensions between frames & masks"
    
    def __len__(self):
        return self.mat_frame.shape[0]
    
    def __getitem__(self, idx):
        return self.mat_frame[idx], self.mat_mask[idx]

###############################################
#  Helper functions to calc distmap parallelly
###############################################

def contour_distmap_parallel(mask):
    neg_mask = 1.0 - mask
    dist = distance_transform_edt(neg_mask) * neg_mask - (distance_transform_edt(mask) - 1) * mask
    return dist

def saw_distmap_parallel(g):

    def _calc_weight_class(g, eps=1e-5):
        """Calculate weights compensating for imbalance class labels"""
        g_augmented = np.zeros_like(g[0]).astype(np.float)
        if g.shape[0] == 1:  # binary mask
            unique_labels = np.unique(g).astype(np.float)
            g_augmented[g[0] == 1] = 1
        else:  # multi-label mask
            unique_labels = np.arange(g.shape[0]).astype(np.float)
            for i in range(g.shape[0]):
                g_augmented[g[i] == 1] = i

        class_score = np.vectorize(lambda x: 1 / np.sqrt((g_augmented == x).sum() + eps))(unique_labels)
        class_score_norm = class_score / (class_score.sum() + eps)
        class_score_dict = dict(zip(unique_labels, class_score_norm))
        weight_class = np.vectorize(lambda x: class_score_dict[x])(g_augmented)

        return weight_class

    def _calc_concave_skeleton(g):
        """Calculate skeleton & contour transformation of the given individual mask"""
        g_convex_hull = convex_hull_image(g)
        g_concave_complement = g_convex_hull - g
        g_skeleton_union = np.bitwise_or(skeletonize(g), skeletonize(g_concave_complement)).astype(np.float)
        g_contour_union = np.bitwise_or(find_boundaries(g), find_boundaries(g_concave_complement)).astype(np.float)
        g_contour_union = dilation(g_contour_union, square(2))

        return g_skeleton_union, g_contour_union

    eps = 1e-5
    sigma = 1

    g_binary = np.zeros((g.shape[1], g.shape[2]))
    if g.shape[0] == 1:  # binary mask
        g_binary[g[0] == 1] = 1.0
    else:  # multi-labek mask
        g_binary[g[0] == 0] = 1.0

    mask_labels = ndi.label(g_binary)[0]
    unique_labels = np.unique(mask_labels)[1:]
    g_indep_masks = []
    for label in unique_labels:
        g_indep_masks.append((mask_labels == label).astype(np.float))
    g_indep_masks = pd.Series(g_indep_masks)
    contour_skeleton = g_indep_masks.apply(_calc_concave_skeleton)
    skeleton_union, contour_union = contour_skeleton.apply(lambda x: x[0]).sum(), contour_skeleton.apply(lambda x: x[1]).sum()
    phi_k = distance_transform_edt(1 - skeleton_union)  # distance to closest skeleton foreground neighbor
    tau = np.max(contour_union * phi_k)  # distance norm factor

    # Sum imbalance class weight & shape-awared weight: (W_saw = W_class + W_shape)
    weight = _calc_weight_class(g, eps=eps) + ndi.gaussian_filter(contour_union * (1 - phi_k / (tau + eps)), sigma=sigma)

    return weight


class DistmapDataLoader(data.Dataset):
    """Precompute distance maps of masks prior to network training"""
    
    def __init__(self, masks, sigma, dist_option):
        super(DistmapDataLoader, self).__init__()
        num_processes = multiprocessing.cpu_count()
        self.pool = multiprocessing.Pool(num_processes)
        if dist_option == 'boundary':
            self.distmap = self.contour_distmap(masks)
        elif dist_option == 'saw':
            self.distmap = self.saw_distmap(masks, sigma)
        elif dist_option == 'dist':
            self.distmap = self.weight_distmap(masks)
        else:  # uniform weight matrix (without distance-based map)
            print("None distmap option provided, returning uniform weighted map...")
            self.distmap = np.ones((len(masks), 1, masks[0].shape[-2], masks[0].shape[-1]))

    def __len__(self):
        return self.distmap.shape[0]
    
    def __getitem__(self, idx):
        return self.distmap[idx]

    def contour_distmap(self, masks):
        """
        Calculate distance map of each pixel to its closest contour in every image
        
        Parameters
        ----------
        masks: array-like
            A 4D array of shape (n_images, channel=1, image_height, image_width),
            where each slice of the matrix along the 0th axis represents one binary mask.
            
        Returns
        -------
        array-like
            A 4D array of shape (n_images, channel=1, image_height, image_width)
        """
        # todo: generalize to multi-class classification
        print('Calculating contour distance maps...')
        """
        distmap = np.zeros_like(masks)
        bar = ChargingBar('Loading', max=len(masks), suffix='%(percent)d%%')
        for i, mask in enumerate(masks):
            bar.next()
            neg_mask = 1.0 - mask
            dist = distance_transform_edt(neg_mask) * neg_mask - (distance_transform_edt(mask) - 1) * mask
            distmap[i, :, :] = dist
        bar.finish()
        """
        distmap = self.pool.map(contour_distmap_parallel, masks)
        self.pool.close()
        self.pool.join()

        distmap = np.stack(distmap, axis=0)

        return distmap

    def weight_distmap(self, masks, w0=10, sigma=5):
        """
        Generate the weight maps as specified in the UNet paper
        for a set of binary masks.
    
        Parameters
        ----------
        masks_list: array-like
            A 4D array of shape (n_images, channel=1, image_height, image_width),
            where each slice of the matrix along the 0th axis represents one binary mask.

        Returns
        -------
        array-like
            A 4D array of shape (n_images, channel=1, image_height, image_width)
        """ 
        # Reference from: https://jaidevd.github.io/posts/weighted-loss-functions-for-instance-segmentation/
        weights = np.zeros_like(masks)
        print('Calculating Weighted distance map...')
        bar = ChargingBar('Loading', max=len(masks), suffix='%(percent)d%%')
        for idx, mask in enumerate(masks):
            bar.next()
            mask = mask.squeeze()
            nrows, ncols = mask.shape
            dist_to_border = distance_transform_edt(1.0 - mask)
            border_loss = w0 * np.exp((-1 * dist_to_border ** 2) / (2 * (sigma ** 2)))
            class_loss = np.zeros((nrows, ncols))
            w_1 = 1 - mask.sum() / class_loss.size
            w_0 = 1 - w_1
            class_loss[mask.sum(0) == 1] = w_1
            class_loss[mask.sum(0) == 0] = w_0
        
            loss = class_loss + border_loss
            weights[idx] = np.expand_dims(loss, axis=0)

        bar.finish()

        return weights

    def saw_distmap(self, masks, sigma, eps=1e-5):
        """
        Shape-awared weighted distance map, highlighting adjacent cell borders, concave cells & smaller cells
        
        Parameters
        ----------
        masks : array-like
            A 4D array of shape (n_images, channel=3, image_height, image_width),
            where each slice of the matrix along the 0th axis represents one binary mask.
        sigma : int
            Standard deviation value for weight-map gaussian filtering
            
        Returns
        -------
        array-like
            A 4D array of shape (n_images, channel=1, image_height, image_width)
        """
        assert masks.ndim == 4, "Invalid masks shape {0}".format(masks.shape)
        print('Calculating shape-awared weight map...')
        weights = self.pool.map(saw_distmap_parallel, masks)
        self.pool.close()
        self.pool.join()

        weights = np.expand_dims(np.stack(weights, axis=0), axis=1)

        return weights


class ImagePreprocessor():
    """
    Preprocessing raw images, remove background noises & smooth regional inhomogeneoous intensity

    steps:
        (1). (optional) Gamma adjustment
        (2). (optional) Background correction
        (3). (optional) Adaptive Histogram Equalization (AHE)
        (4). Rescale intensity to [0,1]

    Returns
    -------
    out : np.ndarray
        Preprocessed image, shape=[1, H, W]
    """

    def __init__(self, img, h, w,
                 gamma=0.5,
                 limit=5.0,
                 grid_size=(16, 16),
                 adjust_gamma=False,
                 dilate=False,
                 enhance=False,
                 cutoff_perc=False):
        """
        Parameters
        ----------
        img : np.ndarray
            Raw image shape=[H, W]
        gamma : np.float
            parameter for gamma adjustment
        limit : np.float
            contrast limit value for AHE
        grid_size : tuple of int
            sliding-window size for AHE
        adjust_gamma : bool
            Whether to perform gamma adjustment (contrast modification)
        dilate : bool
            Whether to perform background correction via g - dilate(g)
        enhance : bool
            Whether to perform Adaptive Histogram Equalization (AHE)
        cutoff_perc : bool
            Whether to threshold the a% lowest pixels (to 0) and b% highest pixels (to 1)
        """
        self.gamma = gamma
        self.limit = limit
        self.height = h
        self.width = w
        self.grid_size = grid_size
        self.adjust_gamma = adjust_gamma
        self.dilate = dilate
        self.enhance = enhance
        self.cutoff_perc = cutoff_perc
        self.img = resize(img, (h, w))

    def _preprocess(self):
        img_out = self.img.copy()
        # img_out = rescale(img_out)
        if is_low_contrast(self.img) or self.adjust_gamma:
            img_out = self._gamma_adjustment(self.img, self.gamma)
        if self.dilate:
            img_out = self._background_correction(img_out)
        if self.enhance:
            img_out = self._ahe(img_out, self.limit, self.grid_size)
        if self.cutoff_perc:
            img_out = self._cutoff_percentile(img_out)
        img_out = rescale(img_out)

        return np.expand_dims(img_out, axis=0)

    @property
    def out(self):
        self._out = self._preprocess()
        assert self._out.ndim == 3 and self._out.shape == (1, self.height, self.width)

        return self._out

    @staticmethod
    def _gamma_adjustment(img, gamma):
        return adjust_gamma(img, gamma)

    @staticmethod
    def _cutoff_percentile(img, low_perc=5, upper_perc=95):
        lower_bound, upper_bound = np.percentile(img, low_perc), np.percentile(img, upper_perc)
        min_intensity, max_intensity = img.min(), img.max()
        img[img < lower_bound] = min_intensity
        img[img > upper_bound] = max_intensity

        return img


    @staticmethod
    def _ahe(img, limit, grid_size):
        """Adjust inhomogeneous intensity distribution via adaptive histogram equalization"""
        img = np.round(img * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid_size)
        img = clahe.apply(img) / 255.0
        threshold = np.percentile(img.squeeze().flatten(), 25)
        img[img <= threshold] = 0

        return img


class MaskPreprocessor():
    """
    Preprocessing masks: Rescaling, Binarization, Class augmentation, etc.

    Returns
    -------
    out : np.ndarray
        Output format (based on "self.option" parameter):
       - binary:
           1-channel binary ground-truth mask, shape=[1, H, W]
           values: 0 = background, 1 = foreground
       - multi:
           3-channel ground-truth mask, shape=[3, H, W]
           Output channels representing 3 classes:
            (1). background
            (2). cell region (nuclei / cytoplasm + nuclei)
            (3). attached border of clotting cells
    """

    def __init__(self, img, h, w, c, image_type='nuclei', option='multi', erode_mask=True, thresh_option='mean'):
        """
        Parameters
        ----------
        img : np.ndarray
            Ground truth masks: shape=[H, W, C=3] or [H, W, C=1]
        image_type : str
            Specification of image type for different 3-channel
            augmentation preprocessing for nuclei / membrane
        erode_mask : bool
            Whether to further take 1-pixel radial erosions
            from the ground-truth mask
        thresh : float
            Global threshold value to binarize mask
            in case that the given mask isn't fully binarized
        """
        self.height = h
        self.width = w
        self.c_in = c
        self.option = option
        self.img = resize(img, (h, w, c))
        self.img_type = image_type
        self.erode_mask = erode_mask
        self.thresh_option = thresh_option

    def _preprocess(self):
        if self.option == 'binary':
            img_out = self._binary_mask()
        elif self.option == 'multi':
            img_out = self._multi_mask()
        else:
            raise NotImplementedError("Unrecognized mask option: {0}".format(self.option))

        return img_out

    def _binary_mask(self):
        """Generate binary ground truth mask"""
        self.img = rescale(self.img)
        if self.c_in == 1:
            img_binary = self._binarize(self.img.squeeze(), erode=self.erode_mask, thresh_option=self.thresh_option)
        else:
            img_binary = self._binarize_multi_channel(self.img.transpose((2, 0, 1)))
        img_out = np.expand_dims(img_binary,  axis=0)

        return img_out

    def _multi_mask(self):
        """Generate multi-class(3) ground truth mask"""
        self.img = rescale(self.img)
        if self.c_in == 1:
            img_binary = self._binarize(self.img.squeeze(), erode=self.erode_mask, thresh_option=self.thresh_option)
        else:
            img_binary = self._binarize_multi_channel(self.img.transpose((2, 0, 1)))

        # Perform different third-label augmentation for nuclei and membrane masks
        if self.img_type == 'nujclei':
            img_processed = self._label_augment(img_binary)
        else:
            img_binary = self._close_border(img_binary)
            img_processed = self._fill_contour(img_binary)

        img_out = self._one_hot_encoding(img_processed, ndim=3)

        return img_out

    @property
    def out(self):
        self._out = self._preprocess()
        if self.option == 'binary':
            assert self._out.shape[0] == 1, "Invalid channel dimension {0} for binary mask".format(self._out.shape[0])
        elif self.option == 'multi':
            assert self._out.shape[0] == 3, "Invalid channel dimension {0} for binary mask".format(self._out.shape[0])

        return self._out

    @staticmethod
    def _label_augment(g):
        """Augment the 3rd class: attaching cell borders """
        se = square(3)
        g_tophat = black_tophat(g, se)
        g_dilation = dilation(g_tophat, se)
        g_aug = g + (np.max(g) + 1) * g_dilation
        g_aug[g_aug == 3.0] = 2.0
        g_aug[find_boundaries(g)] = 2.0  # highlight all borders as the 3rd label

        return g_aug

    @staticmethod
    def _fill_contour(g):
        """Fill in colors for masks highlighting only boundary regions"""
        g_tmp_filled = ndi.binary_fill_holes(g)
        is_boundary = np.bitwise_and(g_tmp_filled == 1.0, g == 1.0)

        g_filled = np.zeros_like(g)
        g_filled[g_tmp_filled == 1.0] = 1.0
        g_filled[is_boundary] = 2.0

        return g_filled

    @staticmethod
    def _close_border(g, limit=32):
        """Close loops with 'half' cells at the edge of the image"""
        upper = np.where(g[0, :] == 1)[0]
        for i in range(len(upper) - 1):
            if upper[i + 1] - upper[i] < limit:
                g[0, upper[i]:upper[i + 1]] = 1

        lower = np.where(g[-1, :] == 1)[0]
        for i in range(len(lower) - 1):
            if lower[i + 1] - lower[i] < limit:
                g[-1, lower[i]:lower[i + 1]] = 1

        left = np.where(g[:, 0] == 1)[0]
        for i in range(len(left) - 1):
            if left[i + 1] - left[i] < limit:
                g[left[i]:left[i + 1], 0] = 1

        right = np.where(g[:, -1] == 1)[0]
        for i in range(len(right) - 1):
            if right[i + 1] - right[i] < limit:
                g[right[i]:right[i + 1], -1] = 1

        return g

    @staticmethod
    def _binarize(g_orig, erode=False, thresh_option='mean'):
        g_orig = rescale(g_orig)  # rescale mask to [0, 1]
        thresh = threshold_mean(g_orig) if thresh_option == 'mean' else threshold_local(g_orig, block_size=15)
        g = (g_orig > thresh).astype(np.float)
        if erode: # Take 1-pixel radial erosion from the raw gt mask
            g = erosion(g, disk(1))

        return g

    @staticmethod
    def _binarize_multi_channel(g_orig):
        """Binarize input masks with 3 channels (storing different information)"""
        g = np.zeros_like(g_orig[0])
        g[g_orig[1] > 0.0] = 1.0
        g[g_orig[2] > 0.0] = 1.0
        g[g_orig[0] > 0.0] = 0.0

        return g

    @staticmethod
    def _one_hot_encoding(g, ndim):
        h, w = g.shape
        g_one_hot = np.zeros((ndim, h, w))
        for i in range(ndim):
            g_one_hot[i, :, :] = (g == i).astype(np.float)

        return g_one_hot

    @staticmethod
    def _inner_distance(mask, indep_masks, alpha=0.8):
        """
        Inner distance transformation to a given gt mask
        """
        def _calc_centroid(mask, val):
            return np.array(np.where(mask == val)).mean(1).astype(int)

        unique_mask_labels = np.unique(indep_masks)
        coords = np.frompyfunc(lambda x: _calc_centroid(indep_masks, x), 1, 1)(unique_mask_labels)
        centroid_coords = tuple(np.stack(np.array(coords)).T)

        centroids = np.zeros_like(mask)
        centroids[centroid_coords] = 1
        inner_dist = (1 / (1 + alpha * ndi.distance_transform_edt(1 - centroids))) * mask

        return inner_dist

    @staticmethod
    def _outer_distance(mask, indep_masks, contours, eps=1e-5):
        """
        Outer distance transformation to a given gt mask
        """
        outer_dist = np.zeros_like(mask)
        for i, idx in enumerate(contours):
            curr_mask = (indep_masks == i + 1).astype(int)
            curr_contour = np.zeros_like(mask)
            idx = idx.squeeze().T
            coords = tuple([idx[1], idx[0]])
            curr_contour[coords] = 1
            dist = ndi.distance_transform_edt(1 - curr_contour) * curr_mask
            outer_dist += dist / (dist.max() + eps)

        return outer_dist

    @staticmethod
    def _label_masks(mask, return_contour=False):
        contours = get_contour(mask)
        mask_indep_labels = np.zeros_like(mask)
        for i in range(len(contours)):
            indep_mask = np.zeros_like(mask)
            contour = contours[i].squeeze(axis=1).astype(np.int32)
            cv2.fillPoly(indep_mask, pts=[contour], color=(255, 255, 255))
            indep_mask /= 255.0
            mask_indep_labels[indep_mask != 0] = i + 1

        return mask_indep_labels, contours if return_contour else mask_indep_labels


def load_data(root_path,
              model_path,
              frame,
              mask,
              image_type='nuclei',
              n_channel_frame=1,
              mask_option='multi',
              height=256,
              width=256,
              sigma=None,
              batch_size=1,
              gamma=0.5,
              limit=None,
              adjust_gamma=False,
              enhance=False,
              dilate=False,
              cutoff_perc=False,
              erode_mask=False,
              thresh_option='mean',
              return_dist=None):
    """
    Load images from directory, preprocess & initialize dataloader object

    Parameters
    ----------
    n_channel_frame : int
        Number of channels for each input image (default: 1)
    image_type : str
        Specification of input image type. Options
         - nuclei
         - membrane
    mask_option : str
        Mask output format:
        'binary' - 1-channel mask,
        'multi'  - 3-channel mask
    sigma : float
        Parameter of gaussian blur parameter for distance map calculation
        (if applying pixel-wise weight maps during loss calculation)
    gamma : float
        Parameter of gamma adjustment parameters for image preprocessing
    limit : float
        Parameter of adaptive histogram equalization for image preprocessing
    adjust_gamma : bool
        Whether to perform gamma adjustment on input images
    enhance : bool
        Whether to enhance input images by performing
        adaptive histogram equalization (AHE)
    dilate : bool
        Whether to perform gaussian smoothing on input images
    cutoff_perc : bool
        Whether to threshold the a% lowest pixels (to 0)
        and b% highest pixels (to 1) on input images
    erode_mask : bool
        Whether to take 1-pixel erosions on ground-truth masks
        (enhance boundary pixels for adjacent cells)
    thresh_option : string
        Threshold option for binarizing output mask preedictions
    return_distance : string
        The option to specify returned distance map;
        Available options: saw, dist, boundary.

    Returns
    -------
    dataloader, distmap : tuple of dataloader of:
        [(image, mask),...] & the corresponding distance-map w.r.t each mask

    """
    # Configure num_channels of ground-truth masks
    if mask_option == 'binary':
        n_channel_mask = 1
    elif mask_option == 'multi':
        n_channel_mask = 3
    else:
        raise NotImplementedError('Unforseen mask option {0}'.format(mask_option))

    frame_path = os.path.join(root_path, frame)
    mask_path = os.path.join(root_path, mask)
    frame_names = sorted(os.listdir(frame_path))
    mask_names = sorted(os.listdir(mask_path))

    assert os.path.exists(frame_path) and os.path.exists(mask_path), "Image directory doesn't exist!!"
    assert return_dist is None or \
           return_dist == 'boundary' or \
           return_dist == 'dist' or \
           return_dist == 'saw', "Unrecognized return_dist option {0}".format(return_dist)

    # Read raw images of frames & masks, store them in np.ndarray
    bar = ChargingBar('Loading', max=len(frame_names), suffix='%(percent)d%%')
    mat_frame = np.zeros((len(frame_names), n_channel_frame, height, width))
    mat_mask = np.zeros((len(mask_names), n_channel_mask, height, width))
    for i in range(len(frame_names)):
        frame_name, mask_name = frame_names[i], mask_names[i]
        bar.next()
        mat_frame[i], mat_mask[i] = read_images(os.path.join(frame_path, frame_name),
                                                os.path.join(mask_path, mask_name),
                                                h=height,
                                                w=width,
                                                image_type=image_type,
                                                gamma=gamma,
                                                limit=limit,
                                                adjust_gamma=adjust_gamma,
                                                enhance=enhance,
                                                dilate=dilate,
                                                cutoff_perc=cutoff_perc,
                                                erode_mask=erode_mask,
                                                thresh_option=thresh_option,
                                                mask_option=mask_option)
    bar.finish()

    dataset = ImageDataLoader(mat_frame, mat_mask)
    distset = DistmapDataLoader(mat_mask, sigma, dist_option=return_dist)
    dataloader = data.DataLoader(dataset, batch_size=batch_size)
    distset = data.DataLoader(distset, batch_size=batch_size)

    return dataloader, distset


def read_images(name1, name2, h, w,
                image_type='nuclei',
                gamma=0.5,
                limit=None,
                adjust_gamma=False,
                enhance=False,
                dilate=False,
                cutoff_perc=False,
                erode_mask=False,
                thresh_option='mean',
                mask_option='multi'):
    """Read and preprocess the images"""
    img_frame_raw = crop_img(cv2.imread(name1, cv2.IMREAD_COLOR))
    img_frame_gray = cv2.cvtColor(img_frame_raw, cv2.COLOR_BGR2GRAY)  # Convert raw image to grayscale
    img_mask_raw = crop_img(cv2.imread(name2, cv2.IMREAD_COLOR))

    # Image & mask preprocessing
    if limit is None:
        limit = 1.0 if 'svg' in name1 else 5.0
    img_frame = ImagePreprocessor(img_frame_gray, h, w,
                                  gamma=gamma,
                                  limit=limit,
                                  adjust_gamma=adjust_gamma,
                                  dilate=dilate,
                                  enhance=enhance,
                                  cutoff_perc=cutoff_perc).out
    n_channel_mask_resize = 3 if 'svg' in name2 else 1
    img_mask = MaskPreprocessor(img_mask_raw, h, w,
                                image_type=image_type,
                                c=n_channel_mask_resize,
                                erode_mask=erode_mask,
                                thresh_option=thresh_option,
                                option=mask_option).out

    return img_frame, img_mask


def crop_img(img):
    """Crop image to have square shape"""
    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w, _ = img.shape
    else:
        raise ValueError('Unrecognized image dimension')
    side_length = min(h, w)
    if w == side_length:
        h_left = h // 2 - side_length // 2
        img_cropped = img[h_left:h_left + side_length, ...]
    else:
        w_left = w // 2 - side_length // 2
        img_cropped = img[:, w_left:w_left + side_length, ...]

    return img_cropped


def rescale(img):
    """Rescale masks, reverse the image if background value exceeds the threshold"""
    img_scaled = rescale_intensity(img, out_range=(0, 1))  # Min-max scale
    # if img_scaled.mean() > threshold:
    #    img_scaled = 1 - img_scaled
    # assert img_scaled.mean() <= 0.5, "ERROR....{0}".format(img_scaled.mean())

    return img_scaled


def top_hat(img, kernel_size=256, scale=False):
    """ top-hat transformation, denoise background, assume input image has shape [C, H, W]"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    img_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    if scale:
        img_tophat = (img_tophat - img_tophat.min()) / (img_tophat.max() - img_tophat.min())  # Min-max scale

    # return img_tophat if img_tophat.ndim == 3 else np.expand_dims(img_tophat, axis=0)
    return img_tophat


def augmentation(root_path, mode='train'):
    """Perform data augmentation based on input images"""
    # configure paths
    frame_path = os.path.join(root_path, mode + '_frames')
    mask_path = os.path.join(root_path, mode + '_masks')
    aug_frame_path = frame_path + '_aug'
    aug_mask_path = mask_path + '_aug'
    aug_tmp_path = os.path.join(root_path, 'augment')
    
    os.makedirs(aug_frame_path, exist_ok=True)
    os.makedirs(aug_mask_path, exist_ok=True)
    
    # Perform augmentation
    p = Augmentor.Pipeline(source_directory=os.path.abspath(frame_path),
                           output_directory=os.path.abspath(aug_tmp_path))
    p.ground_truth(mask_path)
    
    # Apply shift, rotation and elastic deformation
    p.gaussian_distortion(0.3,
    
                          # grid axis for distortion (smaller value --> larger granular distortion)
                          grid_width=3, grid_height=3,
    
                          # magnitude & which corner to distort
                          magnitude=10, corner='bell', method='in')
    
    p.rotate(0.3, max_left_rotation=10, max_right_rotation=10)
    p.flip_random(0.3)
    p.shear(0.2, max_shear_left=5, max_shear_right=5)
    p.skew(0.2, magnitude=0.1)
    
    # Sampling
    sample_size = 10 * len(os.listdir(frame_path))
    p.sample(sample_size)
    
    # Parse augmented images into "frame" & "mask" directories
    aug_names = os.listdir(aug_tmp_path)
    indicator = mode + '_frames_'
    for name in aug_names:
        if name[:len(indicator)] == indicator:  # augmented frame image
            new_name = name.split(indicator + 'original_')[1]
            shutil.copy(os.path.join(aug_tmp_path, name), aug_frame_path)
            os.rename(os.path.join(aug_frame_path, name), os.path.join(aug_frame_path, new_name))
        
        else:  # augmented mask image
            new_name = name.split(indicator)[1]
            shutil.copy(os.path.join(aug_tmp_path, name), aug_mask_path)
            os.rename(os.path.join(aug_mask_path, name), os.path.join(aug_mask_path, new_name))
    
    # Cleanup: remove the temporary directory with original augmented images
    shutil.rmtree(aug_tmp_path)

    # debug: test names
    aug_frame_names = os.listdir(aug_frame_path)
    aug_mask_names = os.listdir(aug_mask_path)
    
    for n1, n2 in zip(aug_frame_names, aug_mask_names):
        assert n1 == n2, "Frame {0} and Mask {1} have inconsistent names".format(n1, n2)
