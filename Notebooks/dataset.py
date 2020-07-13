import os
import shutil
import numpy as np
import cv2
import Augmentor

from progress.bar import ChargingBar
from torch.utils import data
from skimage.transform import resize
from scipy.stats import mode


class ImageDataLoader(data.Dataset):
    
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


def load_data(root_path, frame, mask, n_channel_frame=3, n_channel_mask=1, height=256, width=256, transform=True, return_matrix=False):
    """Load images from directory, preprocess & initialize dataloader object"""
    # Read file names
    frame_path = os.path.join(root_path, frame)
    mask_path = os.path.join(root_path, mask)
    frame_names = sorted(os.listdir(frame_path))
    mask_names = sorted(os.listdir(mask_path))
    
    assert os.path.exists(frame_path) and os.path.exists(mask_path), "Image directory doesn't exist!!"
    
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
    
    if return_matrix:
        return dataset, mat_frame, mat_mask
    else:
        return dataset


def read_images(name1, name2, h, w, transform=True):
    """Read and preprocess the images"""
    img_frame_raw = cv2.imread(name1, cv2.IMREAD_COLOR)
    img_mask_raw = cv2.imread(name2, cv2.IMREAD_COLOR)
    
    # Reshape
    img_frame = resize(img_frame_raw, (h, w, 3))
    img_mask = resize(img_mask_raw, (h, w, 1))
    
    # Axes permutation: (H, W, C) -> (C, H, W), facilitates to PyTorch nn order
    img_frame = img_frame.transpose((2, 0, 1))
    img_mask = img_mask.transpose((2, 0, 1))
    
    # Rescale the frame (raw image), thresholding the mask
    if transform:
        thresh = 0.1 if 'C2-' in name2 else 0.8
        img_frame = top_hat(img_frame)
        # img_mask = border_transform(img_mask, thresh)
        img_mask = rescale(img_mask)  # Inver masks with "1" as background & "0" as segmentation
        img_mask = binarize(img_mask, thresh)
    
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


def top_hat(img, kernel_size=256):
    """ top-hat transformation, denoise background, assume input image has shape [C, H, W]"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    img_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    img_scaled = (img_tophat - img_tophat.min()) / (img_tophat.max() - img_tophat.min())  # Min-max scale
    
    return img_scaled


# https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
def equalize_hist(img):
    """histogram equalization, assume input image has shape [C, H, W]"""
    c1 = np.round(img[0] * 255.0).astype(np.uint8)
    c2 = np.round(img[1] * 255.0).astype(np.uint8)
    c3 = np.round(img[2] * 255.0).astype(np.uint8)
    
    out_c1, out_c2, out_c3 = cv2.equalizeHist(c1), cv2.equalizeHist(c2), cv2.equalizeHist(c3)
    
    img_enhanced = cv2.merge((out_c1, out_c2, out_c3)).transpose((2, 0, 1))
    img_enhanced = img_enhanced / 255.0
    
    return img_enhanced


def calc_weight(mat, threshold=0.0, epsilon=1e-20):
    """Calculate weight for BCE with imbalanced labels"""
    return (mat <= threshold).sum() / ((mat > threshold).sum() + epsilon)


def border_transform(img, threshold=0.5):
    """Retrieve only boundaries of image masks"""
    img = rescale(img.squeeze())
    img = binarize(img)
    
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
        if img[i - 1, W - 2] == img[i - 1, W - 1] == img[i, W - 2] == img[i, W - 1] == img[i + 1][W - 2] == img[
            i + 1, W - 1] == 1:
            out[i, W - 1] = 0
    
    for j in range(1, W - 1):
        if img[0, j - 1] == img[0, j] == img[0, j + 1] == img[1, j - 1] == img[1, j] == img[1, j + 1] == 1:
            out[0, j] = 0
        if img[H - 2, j - 1] == img[H - 2, j] == img[H - 2, j + 1] == img[H - 1, j - 1] == img[H - 1, j] == img[
            H - 1, j + 1] == 1:
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
        assert n1 == n2, "Inconsistent file name between corresponding frame & task"