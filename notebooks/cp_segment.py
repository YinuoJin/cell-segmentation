# Perform 3D segmentation via cellpose

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from mayavi import mlab
from cellpose import models as cp_models
from cellpose import utils as cp_utils
from cellpose import plot as cp_plot


class Cellpose_3d():
    """
    Perform 3d segmentation via cellpose

    Parameters
    ----------
    frames : np.ndarray
        3D input image (D x H x W)

    model_type : str
        Cellpose model type (nuclei / cyto)

    est_diam : bool
        Whether to estimate mask avg. diameter via 2D segmentation

    outdir : str
        Output directory for segmentation results
    """

    def __init__(self, frames, model_type='nuclei', est_diam=True, use_gpu=False, outdir=None, name=None):
        self.frames = frames
        self.n_layers = frames.shape[0]
        self.model_type = model_type
        self.est_diam = est_diam
        self.outdir = outdir
        self.name = name  # output file name
        self.channels = [0, 0]  # by defaul input images are converted to grayscale
        self.cp_model = cp_models.Cellpose(gpu=use_gpu, model_type=model_type)

        # Perform 3D segmentation
        self.res_2d = None
        self.res = self._segment()

    def _estimate_diameter(self):
        """Estimate cell diameter for 3D segmentation by performing 2D segmentation in the middle z-layer"""
        cp_model_2d = cp_models.Cellpose(gpu=True, model_type=self.model_type)
        self.res_2d = cp_model_2d.eval(self.frames[self.n_layers//2], diameter=None, channels=self.channels)

        # cleanup
        del cp_model_2d
        torch.cuda.empty_cache()

        return self.res_2d[-1]

    def _segment(self):
        diam = self._estimate_diameter() if self.est_diam else 15
        res = self.cp_model.eval(self.frames, diameter=diam, do_3D=True, channels=self.channels)
        return res

    @property
    def masks(self):
        return self.res[0].astype(np.uint16)

    @property
    def flows(self):
        return self.res[1]

    @property
    def styles(self):
        return self.res[2]

    def disp_2d_slice(self):
        """Display 2D segmentation results in the mid z-layer"""
        mask, flow, style, diam = self.res_2d
        fig = plt.figure(figsize=(12, 5))
        cp_plot.show_segmentation(fig,
                                  self.frames[self.n_layers//2].astype(np.float),
                                  mask,
                                  flow[0],
                                  channels=self.channels)
        plt.tight_layout()
        plt.show()

    @mlab.show
    def disp_3d_slice(self, axis='z'):
        """Display 3D segmentation results interactively along z-axis or y-axis"""
        mlab.figure(size=(1000, 800))
        res = mlab.volume_slice(self.masks, colormap='jet') if axis == 'z' else mlab.volume_slice(self.masks.transpose((1,2,0)), colormap='jet')
        mlab.axes(line_width=0.5)
        mlab.outline()
        return res

    @mlab.show
    def disp_3d_contour(self):
        mlab.figure(size=(1000, 800))
        res = mlab.contour3d(self.masks.transpose((1,2,0)), contours=10, transparent=True)
        mlab.axes(line_width=0.5)
        mlab.outline()
        return res

    @staticmethod
    def save(masks, outdir, name, to_npy=True, verbose=False):
        """Save segmentation results"""
        # Reshape 3D segmentation results to D x (HxW)
        outdir = os.path.join(outdir, name)
        if verbose:
            print("Saving segmentation result to {}".format(outdir))
        if to_npy:
            np.savetxt(outdir+'.npy', masks)
        else:
            np.savetxt(outdir+'.txt', masks.reshape(self.n_layers, -1))
