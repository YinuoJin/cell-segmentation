import os
import shutil
import numpy as np
import cv2
import Augmentor
import gc

from progress.bar import ChargingBar
from torch.utils import data
from skimage.transform import resize
from skimage.segmentation import find_boundaries
from sklearn.preprocessing import quantile_transform
from skimage.color import rgb2gray
from scipy.stats import mode
from scipy.ndimage.morphology import distance_transform_edt


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
    """Load distance maps of masks"""
    
    def __init__(self, mat_mask, dist_option):
        super(DistmapDataLoader, self).__init__()
        if dist_option == 'boundary':
            self.distmap = self.contour_distmap(mat_mask)
        else:
            self.distmap = self.weight_distmap(mat_mask)

    def __len__(self):
        return self.distmap.shape[0]
    
    def __getitem__(self, idx):
        return self.distmap[idx]

    @staticmethod
    def contour_distmap(masks):
        """Retrieve distance map of each pixel to its closest contour in every image"""
        distmap = np.zeros_like(masks)
        print('Calculating distance maps...')
        bar = ChargingBar('Loading', max=len(masks), suffix='%(percent)d%%')
        for i, mask in enumerate(masks):
            bar.next()
            neg_mask = 1.0 - mask
            dist = distance_transform_edt(neg_mask) * neg_mask - (distance_transform_edt(mask) - 1) * mask
            distmap[i, :, :] = dist
        bar.finish()

        return distmap
    
    @staticmethod
    def weight_distmap(masks_list, w0=10, sigma=5):
        """
        Generate the weight maps as specified in the UNet paper
        for a set of binary masks.
    
        Parameters
        ----------
        masks_list: array-like
            A 4D array of shape (list, channel=1, image_height, image_width),
            where each slice of the matrix along the 0th axis represents one binary mask.

        Returns
        -------
        array-like
            A 2D array of shape (image_height, image_width)
    
        """ 
        # Reference from: https://jaidevd.github.io/posts/weighted-loss-functions-for-instance-segmentation/
        
        weights = np.zeros_like(masks_list)
        print('Calculating distance map...')
        bar = ChargingBar('Loading', max=len(masks_list), suffix='%(percent)d%%')
        for idx, masks in enumerate(masks_list):
            bar.next()
            nrows, ncols = masks.shape[1:]
            masks = (masks > 0).astype(int)
            distMap = np.zeros((nrows * ncols, masks.shape[0]))
            X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
            X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T
        
            for i, mask in enumerate(masks):
                # find the boundary of each mask,
                # compute the distance of each pixel from this boundary
                bounds = find_boundaries(mask, mode='inner')
                X2, Y2 = np.nonzero(bounds)
                xSum = (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2
                ySum = (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
                distMap[:, i] = np.sqrt(xSum + ySum).min(axis=0)

            ix = np.arange(distMap.shape[0])
            if distMap.shape[1] == 1:
                d1 = distMap.ravel()
                border_loss_map = w0 * np.exp((-1 * (d1) ** 2) / (2 * (sigma ** 2)))
            else:
                if distMap.shape[1] == 2:
                    d1_ix, d2_ix = np.argpartition(distMap, 1, axis=1)[:, :2].T
                else:
                    d1_ix, d2_ix = np.argpartition(distMap, 2, axis=1)[:, :2].T
                d1 = distMap[ix, d1_ix]
                d2 = distMap[ix, d2_ix]
                border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))
            
            xBLoss = np.zeros((nrows, ncols))
            xBLoss[X1, Y1] = border_loss_map
            
            # class weight map
            loss = np.zeros((nrows, ncols))
            w_1 = 1 - masks.sum() / loss.size
            w_0 = 1 - w_1
            loss[masks.sum(0) == 1] = w_1
            loss[masks.sum(0) == 0] = w_0
        
            ZZ = xBLoss + loss
            weights[idx] = np.expand_dims(ZZ, axis=0)
            
            # cleanup variables, free memory space
            del distMap, X1, Y1, X2, Y2, xSum, ySum, d1, bounds, border_loss_map, xBLoss, loss, ZZ
            gc.collect()

        bar.finish()

        return weights


def load_data(root_path, frame, mask, n_channel_frame=1, n_channel_mask=1, height=256, width=256, transform=True, return_dist=None):
    """Load images from directory, preprocess & initialize dataloader object"""
    # Read file names
    frame_path = os.path.join(root_path, frame)
    mask_path = os.path.join(root_path, mask)
    frame_names = sorted(os.listdir(frame_path))
    mask_names = sorted(os.listdir(mask_path))
    
    assert os.path.exists(frame_path) and os.path.exists(mask_path), "Image directory doesn't exist!!"
    assert return_dist is None or return_dist == 'boundary' or return_dist == 'weight', "Unrecognized return_dist option"

    # Read raw images of frames & masks, store them in np.ndarray
    mat_frame = np.zeros((len(frame_names), n_channel_frame, height, width))
    mat_mask = np.zeros((len(mask_names), n_channel_mask, height, width))
    
    bar = ChargingBar('Loading', max=len(frame_names), suffix='%(percent)d%%')
    for i, (frame_name, mask_name) in enumerate(zip(frame_names, mask_names)):
        bar.next()
        mat_frame[i], mat_mask[i] = read_images(os.path.join(frame_path, frame_name),
                                                os.path.join(mask_path, mask_name),
                                                height,
                                                width,
                                                transform=transform)
    bar.finish()
    dataset = ImageDataLoader(mat_frame, mat_mask)
    
    if return_dist is not None:
        distset = DistmapDataLoader(mat_mask, return_dist)
        return dataset, distset
    else:
        return dataset


def read_images(name1, name2, h, w, transform=True):
    """Read and preprocess the images"""
    img_frame_raw = cv2.imread(name1, cv2.IMREAD_COLOR)
    img_frame_gray = cv2.cvtColor(img_frame_raw, cv2.COLOR_BGR2GRAY) # Convert raw image to grayscale
    img_mask_raw = cv2.imread(name2, cv2.IMREAD_COLOR)
    
    # Reshape
    img_frame = resize(img_frame_gray, (h, w, 1))
    img_mask = resize(img_mask_raw, (h, w, 1))
    
    # Axes permutation: (H, W, C) -> (C, H, W), facilitates to PyTorch nn order
    img_frame = img_frame.transpose((2, 0, 1))
    img_mask = img_mask.transpose((2, 0, 1))
    
    # Rescale the frame (raw image), thresholding the mask
    if transform:
        # raw image preprocessing
        thresh = 0.1 if 'C2-' in name2 else 0.8
        img_frame = top_hat(img_frame, scale=True)
        #zero_indices = (img_frame < min(img_frame.mean(), 0.1)).squeeze()
        #img_frame = equalize_hist(img_frame, zero_indices=zero_indices)

        # mask preprocessing
        img_mask = rescale(img_mask, threshold=0.5)  # Inverse masks with "1" as background & "0" as segmentation
        img_mask = binarize(img_mask, thresh)
        img_mask = find_boundaries(img_mask, mode='thick').astype(np.float)
            
    return img_frame, img_mask


def rescale(img, threshold=0.5):
    """Rescale masks, reverse the image if "white" is the background"""
    if mode(img, axis=None)[0] > threshold:
        img = img.max() - img
    img_scaled = (img - img.min()) / (img.max() - img.min())  # Min-max scale
    
    return img_scaled


def binarize(img, threshold=0.5):
    """Binarize ground-truth masks"""
    _, img = cv2.threshold(img, threshold, 1.0, cv2.THRESH_BINARY)
    
    return img


def top_hat(img, kernel_size=256, scale=True):
    """ top-hat transformation, denoise background, assume input image has shape [C, H, W]"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    img_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    if scale:
        img_tophat = (img_tophat - img_tophat.min()) / (img_tophat.max() - img_tophat.min())  # Min-max scale

    return img_tophat if img_tophat.ndim == 3 else np.expand_dims(img_tophat, axis=0)


def equalize_hist(img, zero_indices=None):
    """histogram equalization, assume input image has shape [C, H, W]"""
    # reference: # https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
    c = np.round(img[0] * 255.0).astype(np.uint8)
    out = cv2.equalizeHist(c)
    img_enhanced = out / 255.0 
    
    if zero_indices is not None:
        img_enhanced[zero_indices] = 0.0
    
    return np.expand_dims(img_enhanced, axis=0)


def qt(img, zero_indices=None):
    """Quantile transformation to each channnel separately"""
    n_samples = np.min([img.shape[1], 1000])  # set n_quantiles to n_samples if n_samples < 1000
    
    img_transformed = np.zeros_like(img)
    for i in range(img.shape[0]):
        img_transformed[i] = quantile_transform(img[i], n_quantiles=n_samples)
        if zero_indices is not None:
            img_transformed[i][zero_indices] = 0.0
    
    return img


def calc_weight(mat, threshold=0.0, epsilon=1e-20):
    """Calculate weight for BCE with imbalanced labels"""
    return (mat <= threshold).sum() / ((mat > threshold).sum() + epsilon)


def border_transform(img):
    """Retrieve only boundaries of image masks"""
    img = img.squeeze()
    H, W = img.shape
    out = img.copy()
    
    # Corner
    if img[0, 0] == img[0, 1] == img[1, 0] == img[1, 1] == 1:
        out[0, 0] = 0
    if img[0, W - 2] == img[0, W - 1] == img[1, W - 2] == img[1, W - 1] == 1:
        out[0, W - 1] = 0
    if img[H - 2, 0] == img[H - 2, 1] == img[H - 1, 0] == img[H - 1, 1] == 1:
        out[H - 1, 0] = 0
    if img[H - 2, W - 2] == img[H - 2, W - 1] == img[H - 1, W - 2] == img[H - 1, W - 1] == 1:
        out[H - 1, W - 1] = 0
    
    # Side
    for i in range(1, H - 1):
        if img[i - 1, 0] == img[i - 1, 1] == img[i, 0] == img[i, 1] == img[i + 1, 0] == img[i + 1, 1] == 1:
            out[i, 0] = 0
        if img[i - 1, W - 2] == img[i - 1, W - 1] == img[i, W - 2] == img[i, W - 1] == img[i + 1][W - 2] == img[i + 1, W - 1] == 1:
            out[i, W - 1] = 0
    
    for j in range(1, W - 1):
        if img[0, j - 1] == img[0, j] == img[0, j + 1] == img[1, j - 1] == img[1, j] == img[1, j + 1] == 1:
            out[0, j] = 0
        if img[H - 2, j - 1] == img[H - 2, j] == img[H - 2, j + 1] == img[H - 1, j - 1] == img[H - 1, j] == img[H - 1, j + 1] == 1:
            out[H - 1, j] = 0
    
    # Inner regions
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if img[i - 1, j - 1] == img[i - 1, j] == img[i - 1, j + 1] == \
               img[i, j - 1] == img[i, j] == img[i, j + 1] == \
               img[i + 1, j - 1] == img[i + 1, j] == img[i + 1, j + 1] == 1:
                out[i, j] = 0
    
    return np.expand_dims(out, axis=0)


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
    p.gaussian_distortion(1,
    
                          # grid axis for distortion (smaller value --> larger granular distortion)
                          grid_width=3, grid_height=3,
    
                          # magnitude & which corner to distort
                          magnitude=50, corner='bell', method='in')
    
    p.rotate(1, max_left_rotation=20, max_right_rotation=20)
    p.flip_random(0.5)
    p.shear(0.5, max_shear_left=15, max_shear_right=15)
    p.skew(0.5, magnitude=0.2)
    
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

