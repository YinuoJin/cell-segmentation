#!/usr/bin/env python
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from argparse import RawTextHelpFormatter
from progress.bar import ChargingBar
from torch.utils import data
from scipy.spatial.distance import directed_hausdorff
from dataset import load_data, augmentation, calc_weight
from model import Unet
from utils import IoULoss, SoftDiceLoss, SurfaceLoss, DistWeightBCELoss


def run_one_epoch(model, dataloader, distmap, optimizer, loss_fn, train=False, device=None):
    """Single epoch training/validating"""
    torch.set_grad_enabled(train)
    model.train() if train else model.eval()
    losses = []
    accuracies = [] 

    if train:
        bar = ChargingBar('Train', max=len(dataloader), suffix='%(percent)d%%')
    else:
        bar = ChargingBar('Valid', max=len(dataloader), suffix='%(percent)d%%')
    
    if distmap is None: # loss functions without distmap
        for dp in dataloader:
            bar.next()
            x, y = dp
            x, y = x.float(), y.float() # Type conversion (avoid bugs of DOUBLE <==> FLOAT)
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(y, output)
        
            if train:  # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.detach().cpu().numpy())

            # tmp: assign hausdorff distance as "accuracy"
            hd_list = []
            for r1, r2 in zip(y, output):
                r1 = r1.detach().cpu().squeeze().numpy()
                r2 = r2.detach().cpu().squeeze().numpy()
                hd = max(directed_hausdorff(r1, r2)[0], directed_hausdorff(r2, r1)[0])
                hd_list.append(hd)

            accuracies.append(np.mean(hd_list))
            #accuracy = torch.mean(((output > 0.5) == (y > 0.5)).float())
            #accuracies.append(accuracy.detach().cpu().numpy())

    else:  # loss functions with distmap (bce, boundary)
        for dp, dist in zip(dataloader, distmap):
            bar.next()
            x, y = dp
            x, y, dist = x.float(), y.float(), dist.float() # Type conversion (avoid bugs of DOUBLE <==> FLOAT)
            x, y, dist = x.to(device), y.to(device), dist.to(device)
            output = model(x)
            loss = loss_fn(y, output, dist)
        
            if train:  # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.detach().cpu().numpy())

            # tmp: assign hausdorff distance as "accuracy"
            hd_list = []
            for r1, r2 in zip(y, output):
                r1 = r1.detach().cpu().squeeze().numpy()
                r2 = r2.detach().cpu().squeeze().numpy()
                hd = max(directed_hausdorff(r1, r2)[0], directed_hausdorff(r2, r1)[0])
                hd_list.append(hd)

            accuracies.append(np.mean(hd_list))
            #accuracy = torch.mean(((output > 0.5) == (y > 0.5)).float())
            #accuracies.append(accuracy.detach().cpu().numpy())
    bar.finish()
    
    return np.mean(losses), np.mean(accuracies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet training options',
            formatter_class=RawTextHelpFormatter)
    parser.add_argument('-i', '--root-path', dest='root_path', type=str, default=None, action='store',
                        help='Root directory of input image datasets for training')
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=8, action='store',
                        help='Batch size')
    parser.add_argument('-l', '--loss', dest='loss', type=str, default='bce', action='store',
                        help='Loss function option: (1). bce; (2). jaccard; (3).dice; (4).boundary')
    parser.add_argument('-n', '--n-epochs', dest='n_epochs',  type=int, default=150, action='store',
                        help='Total number of epoches for training')
    parser.add_argument('-r', '--loss-rate', dest='lr', type=float, default=0.01, action='store',
                        help='Loss rate')
    parser.add_argument('-p', '--patience', dest='patience_counter', type=int, default=30, action='store',
                        help='Patience counter for early-stopping or lr-tuning')
    parser.add_argument('-a', '--augment', dest='augment', action='store_true',
                        help='Whether to perform data augmentation in the current run')
    parser.add_argument('--early-stop', dest='early_stop', action='store_true', 
                        help='Whether to perform early-stopping; If False, lr is halved when reaching each patience')
    parser.add_argument('--region-option', dest='region_option', action='store_true',
                        help='Whether to use dice loss as the Region-based loss for boundary loss; If False, jaccard loss is used instead')
    parser.add_argument('--enhance-img', dest='enhance_img', action='store_true',
                        help='Whether to use Quantile transformation / Equalization normalization to enhance raw images')
    parser.add_argument('--contour-mask', dest='contour', action='store_true',
                        help='Whether to take contours of ground-truth masks')

    parser.set_defaults(feature=True)
    args = parser.parse_args()

    # Parse arguments
    root_path = '../datasets/multi_cell_custom_data_without_shapes/' if args.root_path is None else args.root_path
    augment = args.augment
    n_epochs = args.n_epochs
    lr = args.lr
    batch_size = args.batch_size
    patience_counter = args.patience_counter
    early_stop = args.early_stop
    loss = args.loss
    region_option = args.region_option
    enhance = args.enhance_img
    contour = args.contour
    dist = None  # distance map option for weighted BCE loss or boundary loss

    if loss == 'bce':
        loss_fn = DistWeightBCELoss()
        dist = 'weight'
    elif loss == 'jaccard':
        loss_fn = IoULoss()
    elif loss == 'dice':
        loss_fn = SoftDiceLoss()
    elif loss == 'boundary':
        alpha = 1.0  # alpha value for Boundary loss --> (a * Region-based loss + (1-a) * boundary loss)
        dist = 'boundary'
    else:
        raise NotImplementedError('Loss function not recognized, available options: (1).bce; (2).jaccard; (3).dice; (4).boundary')

    # data augmentation on training & validation sets
    # augmented images are saved to file, no need to run the following code in every run
    if augment:
        print('Performing data augmentation...')
        augmentation(root_path)
        augmentation(root_path, mode='val')
    
    # load dataset
    print('Loading datasets...')
    print('- Training set:')
    train_dataset, train_distmap = load_data(root_path, 'train_frames_aug', 'train_masks_aug', enhance=enhance, contour=contour, return_dist=dist)
    print('- Validation set:')
    val_dataset, val_distmap = load_data(root_path, 'val_frames_aug', 'val_masks_aug', enhance=enhance, contour=contour, return_dist=dist)
    # print('- Test set':')
    # test_dataset = load_data(root_path, 'test_frames', 'test_masks')
    
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size)
    # test_dataloader = data.DataLoader(test_dataset, batch_size=1)
    if train_distmap is not None:
        train_distmap = data.DataLoader(train_distmap, batch_size=batch_size)
        val_distmap = data.DataLoader(val_distmap, batch_size=batch_size)
 
    # Initialize network & training, transfer to GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Unet(1)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, amsgrad=True)

    train_accs, train_losses, val_accs, val_losses = [], [], [], []
    best_val_loss = np.inf
    check_point_filename = 'model_checkpoint.pt'
     
    # training
    print('Training the network...')
    for epoch in range(n_epochs):
        t0 = time.perf_counter()
        if loss == 'boundary':
            loss_fn = SurfaceLoss(alpha=alpha, dice=region_option)
            alpha -= 0.005 if alpha > 0.75 else alpha
        
        print('*------------------------------------------------------*')
        train_loss, train_acc = run_one_epoch(net, train_dataloader, train_distmap, optimizer, loss_fn, train=True, device=device)
        val_loss, val_acc = run_one_epoch(net, val_dataloader, val_distmap, optimizer, loss_fn, device=device)
        
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), check_point_filename)
        else:
            patience_counter -= 1
    
        if patience_counter <= 0:
            if early_stop: # early stopping
                break
            else: # halved lr every time patience limit is reached
                patience_counter = 30
                lr /= 2
                optimizer = torch.optim.Adam(net.parameters(), lr=lr, amsgrad=True)
        
        delta_t = time.perf_counter() - t0
        print("Epoch %i took %.2fs. Train loss: %.4f acc: %.4f. Val loss: %.4f acc: %.4f. Patience left: %i" %
              (epoch + 1, delta_t, train_loss, train_acc, val_loss, val_acc, patience_counter))

    # Save accuracies & losses
    out_path = '../results/'
    os.makedirs(out_path, exist_ok=True)
    train_accs = np.array(train_accs)
    train_losses = np.array(train_losses)
    val_accs = np.array(val_accs)
    val_losses = np.array(val_losses)

    np.savetxt(out_path + 'acc_train.txt', train_accs)
    np.savetxt(out_path + 'loss_train.txt', train_losses)
    np.savetxt(out_path + 'acc_val.txt', val_accs)
    np.savetxt(out_path + 'loss_val.txt', val_losses)
