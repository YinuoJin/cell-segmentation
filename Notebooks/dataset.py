import os
import shutil
import numpy as np
import cv2
import Augmentor
import gc

from progress.bar import ChargingBar
from torch.utils import data
from scipy.stats import mode
from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt
from skimage.exposure import adjust_gamma, is_low_contrast, match_histograms, rescale_intensity
from skimage.morphology import black_tophat, convex_hull_image, skeletonize, square, disk, dilation, reconstruction
from skimage.filters import threshold_mean
from skimage.transform import resize
from skimage.segmentation import find_boundaries

from utils import get_contour


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


class DistmapDataLoader(data.Dataset):
    """Precompute distance maps of masks prior to network training"""
    
    def __init__(self, masks, sigma, dist_option):
        super(DistmapDataLoader, self).__init__()
        if dist_option == 'boundary':
            self.distmap = self.contour_distmap(masks)
        elif dist_option == 'saw':
            self.distmap = self.saw_distmap(masks, sigma)
        else:
            self.distmap = self.weight_distmap(masks)

    def __len__(self):
        return self.distmap.shape[0]
    
    def __getitem__(self, idx):
        return self.distmap[idx]

    @staticmethod
    def contour_distmap(masks):
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
        distmap = np.zeros_like(masks)
        print('Calculating contour distance maps...')
        bar = ChargingBar('Loading', max=len(masks), suffix='%(percent)d%%')
        for i, mask in enumerate(masks):
            bar.next()
            neg_mask = 1.0 - mask
            dist = distance_transform_edt(neg_mask) * neg_mask - (distance_transform_edt(mask) - 1) * mask
            distmap[i, :, :] = dist
        bar.finish()

        return distmap
    
    @staticmethod
    def weight_distmap(masks, w0=10, sigma=5):
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
    
    @staticmethod
    def saw_distmap(masks, sigma):
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
        def _calc_weight_class(g, smooth=1e-10):
            """Calculate weights compensating for imbalance class labels"""
            weight_class = np.zeros((g.shape[1], g.shape[2]))
            w = np.array([1 / (np.sum(g[0]) + smooth), 1 / (np.sum(g[1]) + smooth), 1 / (np.sum(g[2]) + smooth)])
            w_norm = w / w.sum()
            for i in range(3):
                weight_class[g[i] == 1] = w_norm[i]
        
            return weight_class
        
        def _create_concave_complement(g):
            """create concave complement masks w.r.t. original masks"""
            # Generate n 2D arrays, separating individual masks from each other
            # Generate concave complement w.r.t. to each original mask, stack concaves into single 2D array
            contours = get_contour(g)
            g_concave_comp = np.zeros((g.shape[0], g.shape[1]))
            for i in range(len(contours)):
                g_indep_mask = np.zeros_like(g_concave_comp)
                contour = contours[i].squeeze(axis=1).astype(np.int32)
                cv2.fillPoly(g_indep_mask, pts=[contour], color=(255, 255, 255))
                g_indep_mask /= 255.0
                g_convex = convex_hull_image(g_indep_mask)
                g_concave_comp[g_convex - g_indep_mask > 0] = 1.0
        
            return g_concave_comp
    
        assert masks.ndim == 4 and masks.shape[1] == 3, "Invalid masks shape {0}".format(masks.shape)
        print('Calculating shape-awared weight map...')
        
        if sigma is None:
            sigma = 4
        bar = ChargingBar('Loading', max=len(masks), suffix='%(percent)d%%')
        weights = np.zeros((masks.shape[0], 1, masks.shape[2], masks.shape[3]))
        for i, g in enumerate(masks):
            bar.next()
            g_binary = np.zeros((g.shape[1], g.shape[2]))
            g_binary[g[1] == 1.0] = 1.0
            g_concave = _create_concave_complement(g_binary)
            skeleton_union = np.bitwise_or(skeletonize(g_binary), skeletonize(g_concave)).astype(np.float)
            contour_union = np.bitwise_or(find_boundaries(g_binary), find_boundaries(g_concave)).astype(np.float)
    
            phi_k = distance_transform_edt(1 - skeleton_union)  # distance to closest skeleton foreground neighbor
            tau = np.max(contour_union * phi_k)  # distance norm factor
    
            # Sum imbalance class weight & shape-awared weight: (W_saw = W_class + W_shape)
            weight = _calc_weight_class(g) + ndi.gaussian_filter(contour_union * (1 - phi_k / tau), sigma=sigma)
            weight_scaled = weight * 10
            weights[i] = np.expand_dims(weight_scaled, axis=0)
            
        bar.finish()
            
        return weights
        
        
def load_data(root_path,
              frame, mask,
              n_channel_frame=1,
              n_channel_mask=3,
              height=256,
              width=256,
              sigma=None,
              limit=None,
              enhance=False,
              dilate=False,
              return_dist=None):
    """Load images from directory, preprocess & initialize dataloader object"""
    # Read file names
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
                                                height,
                                                width,
                                                limit,
                                                n_channel_mask == 1,
                                                dilate,
                                                enhance)
        
    bar.finish()
    dataset = ImageDataLoader(mat_frame, mat_mask)
    distset = None if return_dist is None else DistmapDataLoader(mat_mask, sigma, return_dist)
    
    return dataset, distset


def read_images(name1, name2, h, w, limit=None, binary_mask=False, dilate=False, enhance=False):
    """Read and preprocess the images"""
    img_frame_raw = cv2.imread(name1, cv2.IMREAD_COLOR)
    img_frame_gray = cv2.cvtColor(img_frame_raw, cv2.COLOR_BGR2GRAY)  # Convert raw image to grayscale
    img_mask_raw = cv2.imread(name2, cv2.IMREAD_COLOR)
    
    # Raw image preprocessing
    if limit is None:
        limit = 1.0 if 'svg' in name1 else 5.0
    img_frame = resize(img_frame_gray, (h, w))
    img_frame = img_preprocessing(img_frame, limit=limit, dilate=dilate, enhance=enhance)
    
    # Mask preprocessing, thresholding & label augmentation
    img_mask = resize(img_mask_raw, (h, w, 3)) if 'svg' in name2 and not binary_mask else resize(img_mask_raw, (h, w, 1))
    img_mask = mask_preprocessing(img_mask, binary_mask)
    
    # Check dimensions
    if binary_mask:
        assert img_frame.shape == (1, h, w) and img_mask.shape == (1, h, w), 'Invalid image shape!'
    else:
        assert img_frame.shape == (1, h, w) and img_mask.shape == (3, h, w), 'Invalid image shape!'
        
    return img_frame, img_mask


def img_preprocessing(img, limit=1.0, grid_size=(16, 16), dilate=False, enhance=True):
    """
    Preprocessing raw images, remove background noises & smooth regional inhomogeneoous intensity
    
    steps:
        (1). (optional) Gamma adjustment
        (2). (optional) Background correction
        (3). Adaptive Histogram Equalization (AHE)
        (4). Rescale intensity to [0,1]
    
    Parameters
    ----------
    img : np.ndarray
        Raw image shape=[H, W]
    limit : np.float
        contrast limit value for AHE
    grid_size : tuple of int
        sliding-window size for AHE
    dilate : bool
        Whether to perform background correction via g - dilate(g)
    """
    # Gamma adjustment
    img = adjust_gamma(img, 0.5) if is_low_contrast(img) else img
    
    # Background correction with morphologial operations (g - dilation(g))
    if dilate:
        seed = img.copy()
        seed[1:-1, 1:-1] = img.min()
        dilated = reconstruction(seed, img, method='dilation')
        img = img - dilated
        
    # AHE
    if enhance: # further separate foreground & background contrast
        img = np.round(img * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid_size)
        img = clahe.apply(img) / 255.0

    # Rescale intensity
    img = rescale_intensity(img, out_range=(0, 1))
    
    return np.expand_dims(img, axis=0)


def mask_preprocessing(img, binary_mask=False):
    """
    Preprocessing masks: Rescaling, Binarization, Class augmentation

    Output channels representing 3 classes:
        (1). background
        (2). cell region (cytoplasm + nuclei)
        (3). attached border of clotting cells

    Parameters
    ----------
    img : np.ndarray
        Ground truth masks: shape=[H, W, 3] (if from from annotated fluroscence datasets) or [H, W, 1]
    binary_mask : bool
        Whether return binarized mask or one-hot encoded (3-channel) mask (default=False)

    Returns
    -------
    img_one_hot : np.ndarray
        one-hot encoded ground truth mask, shape=[3, H, W]
    """
    
    def _label_augment(g):
        """Augment the 3rd class: attaching cell borders """
        se = square(3)
        g_tophat = black_tophat(g, se)
        g_dilation = dilation(g_tophat, se)
        g_aug = g + (np.max(g) + 1) * g_dilation
        g_aug[g_aug == 3.0] = 2.0
        
        # debug: try highlighting all borders as the 3rd label
        g_aug[find_boundaries(g)] = 2.0
        
        return g_aug
    
    def _fill_boundary_mask(g):
        """Create filled-in masks for original boundary masks"""
        g = _binarize(rescale(g))
        g = dilation(g, disk(1))  # enhance cell borders
        g_tmp_filled = ndi.binary_fill_holes(g)
        is_boundary = np.bitwise_and(g_tmp_filled == 1.0, g == 1.0)
        
        g_filled = np.zeros_like(g)
        g_filled[g_tmp_filled == 1.0] = 1.0
        g_filled[is_boundary] = 2.0
        # dist_to_background = ndi.distance_transform_edt(g_tmp_filled)
        # g_filled[dist_to_background <= 2.0] = 0.0
        
        return g_filled
    
    def _binarize_multi_channel(g_orig):
        g = np.zeros_like(g_orig[0])
        g[g_orig[1] > 0.0] = 1.0
        g[g_orig[2] > 0.0] = 1.0
        g[g_orig[0] > 0.0] = 0.0
        
        return g
    
    def _binarize(g_orig):
        thresh = threshold_mean(g_orig)
        g = (g_orig > thresh).astype(np.float)
        
        return g
    
    def _one_hot_encoding(g):
        h, w = g.shape
        g_one_hot = np.zeros((3, h, w))
        for i in range(3):
            g_one_hot[i, :, :] = (g == i).astype(np.float)
        
        return g_one_hot
    
    if binary_mask:
        img_out = _binarize(rescale(img))
        img_out = np.expand_dims(dilation(img_out.squeeze(), disk(1)), axis=0)
    else:
        if img.mean() > 0.5:
            img_processed = _fill_boundary_mask(img.squeeze())
        else:
            img_binary_raw = _binarize(img.squeeze()) if img.shape[2] == 1 else _binarize_multi_channel(
                img.transpose((2, 0, 1)))
            img_processed = _label_augment(img_binary_raw)
        img_out = _one_hot_encoding(img_processed)
    
    return img_out
    
    
def rescale(img, threshold=0.5):
    """Rescale masks, reverse the image if background value exceeds the threshold"""
    if mode(img, axis=None)[0] > threshold:
        img = img.max() - img
    img_scaled = (img - img.min()) / (img.max() - img.min())  # Min-max scale
    
    return img_scaled


def top_hat(img, kernel_size=256, scale=True):
    """ top-hat transformation, denoise background, assume input image has shape [C, H, W]"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    img_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    if scale:
        img_tophat = (img_tophat - img_tophat.min()) / (img_tophat.max() - img_tophat.min())  # Min-max scale

    return img_tophat if img_tophat.ndim == 3 else np.expand_dims(img_tophat, axis=0)


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
