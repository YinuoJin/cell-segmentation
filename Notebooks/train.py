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
from dataset import load_data, augmentation
from model import Unet
from utils import hausdorff, calc_accuracy_score, calc_f1_score, IoULoss, SoftDiceLoss, SurfaceLoss, ShapeBCELoss


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
            x, y = x.float(), y.float()  # Type conversion
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(y, output)
        
            if train:  # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.detach().cpu().numpy())

            # hausdorff distance as "accuracy"
            # distance = hausdorff(y, output)
            # accuracies.append(distance)

            # binary classification accuracy
            # accuracy = torch.mean(((output > 0.5) == (y > 0.5)).float())
            # accuracies.append(accuracy.detach().cpu().numpy())

            # multi-label classification F1-score
            accuracies.append(calc_f1_score(y, output))  # multi-label accuracy

    else:  # loss functions with distmap (bce, boundary, saw)
        for dp, dist in zip(dataloader, distmap):
            bar.next()
            x, y = dp
            x, y, dist = x.float(), y.float(), dist.float()  # Type conversion (avoid bugs of DOUBLE <==> FLOAT)
            x, y, dist = x.to(device), y.to(device), dist.to(device)
            output = model(x)
            loss = loss_fn(y, output, dist)
        
            if train:  # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.detach().cpu().numpy())

            # hausdorff distance as "accuracy"
            # distance = hausdorff(y, output)
            # accuracies.append(distance)

            # binary classification accuracy
            # accuracy = torch.mean(((output > 0.5) == (y > 0.5)).float())
            # accuracies.append(accuracy.detach().cpu().numpy())
            
            # multi-label classification F1-score
            accuracies.append(calc_f1_score(y, output))  # multi-label accuracy

            del x, y, dist
            torch.cuda.empty_cache()

    bar.finish()
    
    return np.mean(losses), np.mean(accuracies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet training options',
            formatter_class=RawTextHelpFormatter)
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', dest='root_path', type=str, required=True, action='store',
                        help='Root directory of input image datasets for training')
    
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-b', dest='batch_size', type=int, default=4, action='store',
                        help='Batch size')
    optional.add_argument('-c', dest='channel', type=int, default=3, action='store',
                        help='Output channel size')
    optional.add_argument('-l', dest='loss', type=str, default='bce', action='store',
                        help='Loss function\n  Options: (1). bce; (2). jaccard; (3).dice; (4).boundary'),
    optional.add_argument('-n', dest='n_epochs',  type=int, default=150, action='store',
                        help='Total number of epoches for training')
    optional.add_argument('-r', dest='lr', type=float, default=0.01, action='store',
                        help='Learning rate')
    optional.add_argument('-p', dest='patience_counter', type=int, default=30, action='store',
                        help='Patience counter for early-stopping or lr-tuning')
    optional.add_argument('--augment', dest='augment', action='store_true',
                        help='Whether to perform data augmentation in the current run')
    optional.add_argument('--early-stop', dest='early_stop', action='store_true',
                        help='Whether to perform early-stopping; If False, lr is halved when reaching each patience')
    optional.add_argument('--region-option', dest='region_option', action='store_true',
                        help='Whether to use dice loss as the Region-based loss for boundary loss; If False, jaccard loss is used instead')

    parser.set_defaults(feature=True)
    args = parser.parse_args()

    # Parameter initialization
    root_path = args.root_path
    augment = args.augment
    n_epochs = args.n_epochs
    lr = args.lr
    batch_size = args.batch_size
    n_output_channels = args.channel
    patience_counter = args.patience_counter
    early_stop = args.early_stop
    loss = args.loss
    region_option = args.region_option
    dist = None  # weighted distmap indicator
    
    # data augmentation on training
    if augment:
        print('Performing data augmentation...')
        augmentation(root_path, mode='train')
        augmentation(root_path, mode='val')
        augmentation(root_path, mode='test')
        exit()
        
    if loss == 'bce':
        dist = 'dist' if n_output_channels == 1 else 'saw'
        loss_fn = ShapeBCELoss()
    elif loss == 'jaccard':
        loss_fn = IoULoss()
    elif loss == 'dice':
        loss_fn = SoftDiceLoss()
    elif loss == 'boundary':
        alpha = 1.0  # a: (a * Region-based loss + (1-a) * boundary loss)
        dist = 'boundary'
        loss_fn = SurfaceLoss(alpha=alpha, dice=region_option)
    else:
        raise NotImplementedError('Loss function {0} not recognized'.format(loss))
    
    sigma = None
    if dist == 'saw': sigma = 3 if 'nuclei' in root_path else 1
    
    # load dataset
    print('Loading datasets...')
    print('- Training set:')
    train_dataset, train_distmap = load_data(root_path, 'train_frames', 'train_masks',
                                             n_channel_mask=n_output_channels, sigma=sigma,
                                             return_dist=dist)
    print('- Validation set:')
    val_dataset, val_distmap = load_data(root_path, 'val_frames', 'val_masks',
                                         n_channel_mask=n_output_channels, sigma=sigma,
                                         return_dist=dist)
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
    net = Unet(1, n_output_channels)
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
            loss_fn.alpha = alpha
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
            else:  # halved lr every time patience limit is reached
                patience_counter = args.patience_counter
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
