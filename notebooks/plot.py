#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plo
import seaborn as sns
import argparse

from argparse import RawTextHelpFormatter
from matplotlib import colors
from mayavi import mlab
from cellpose import utils as cp_utils
from cellpose import plot as cp_plot


def recover_3d(file_name):
    img = np.loadtxt(file_name).astype(np.float)
    n_layers, npos = img.shape
    img = img.reshape(n_layers, int(np.sqrt(npos)), -1)
    
    return img


#################################
# 2D plot
#################################

# to be implemented


#################################
# 3D plot
#################################

def plot_3d(file_name):
    """Visualize 3D segmentation results in mayavi from saved numpy file"""
    try:
        img_3d = recover_3d(file_name)
        assert img_3d.ndim == 3, "Invalid dimension of 3D image: {}".format(img_3d.ndim)
        mlab.contour3d(img_3d.transpose((1,2,0)), contours=10, transparent=True)
    except FileNotFoundError:
        print("3D segmentation array loading unsuccessful")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize segmentation results',
            formatter_class=RawTextHelpFormatter)
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')

    parser.add_argument('-i', dest='root_path', type=str, default='./', required=True, action='store',
            help='Root directory of input image')
    parser.add_argument('-f', dest='file_name', type=str, required=True, action='store',
            help='Root directory of file name')

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--3d', dest='is_3d', action='store_true',
            help='Visualize 3D segmentation')

    #todo:  addition options tbd

    args = parser.parse_args()
    file_name = os.path.join(args.root_path, args.file_name)

    # tmp
    if args.is_3d:
        plot_3d(file_name)


