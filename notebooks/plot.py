#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
import argparse

from argparse import RawTextHelpFormatter
from cellpose import utils as cp_utils
from cellpose import plot as cp_plot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize segmentation results',
            formatter_class=RawTextHelpFormatter)
    parser.add_argument('-f', dest='file_name', type=str, required=True, action='store',
            help='Root directory of file name')
    parser.add_argument('--axis', dest='axis', type=str, default='z', action='store',
            help='Axis to move along (for slice plot)')
    #parser.add_argument('--3d', dest='is_3d', action='store_true',
    #        help='Visualize 3D segmentation')
    parser.add_argument('--slice', dest='slice', action='store_true',
            help='Visualize 2D slices along z-axis')
    
    args = parser.parse_args()


#-------------------------
# Mayavi 3d plotting
#-------------------------

from mayavi import mlab

@mlab.show
def plot_3d(img, slice=False, axis='z'):
    """Visualize 3D segmentation results in mayavi from saved numpy file"""
    try:
        assert img.ndim == 3, "Invalid dimension of 3D image: {}".format(img_3d.ndim)
        mlab.figure(size=(1000, 800))
        if slice:
            res = mlab.volume_slice(img, colormap='jet') if axis == 'z' else mlab.volume_slice(img.transpose((1,2,0)), colormap='jet')
        else:
            res = mlab.contour3d(img.transpose((1,2,0)), contours=10, transparent=True)
        mlab.axes(line_width=0.5)
        mlab.outline()
    except FileNotFoundError:
        print("3D segmentation array loading unsuccessful")

    return res


if __name__ == '__main__':
    img = recover_3d
    plot_3d(args.file_name, slice=args.slice, axis=args.axis)
