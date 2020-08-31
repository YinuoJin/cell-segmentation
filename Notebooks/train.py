#!/usr/bin/env python
import os
import time
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import argparse

from argparse import RawTextHelpFormatter
from progress.bar import ChargingBar
from torch.utils import data
from dataset import load_data, augmentation
from model import Unet
from postprocessing import Postprocessor
from utils import hausdorff, calc_accuracy_score, calc_f1_score, IoULoss, SoftDiceLoss, SurfaceLoss, ShapeBCELoss


def train(root_path, bs, lr, pc, mask_option, dist, sigma, loss_fn, alpha):
    print('Loading datasets...')
    print('- Training set:')
    train_dataset, train_distmap = load_data(root_path, 'train_frames', 'train_masks',
                                             mask_option=mask_option, sigma=sigma,
                                             return_dist=dist)
    print('- Validation set:')
    val_dataset, val_distmap = load_data(root_path, 'val_frames', 'val_masks',
                                         mask_option=mask_option, sigma=sigma,
                                         return_dist=dist)
    train_dataloader = data.DataLoader(train_dataset, batch_size=bs)
    val_dataloader = data.DataLoader(val_dataset, batch_size=bs)
    if train_distmap is not None:
        train_distmap = data.DataLoader(train_distmap, batch_size=bs)
        val_distmap = data.DataLoader(val_distmap, batch_size=bs)

    # Initialize network & training, transfer to GPU is available
    c_out = 1 if mask_option == 'binary 'else 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Unet(1, c_out)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, amsgrad=True)

    train_accs, train_losses, val_accs, val_losses = [], [], [], []
    best_val_loss = np.inf
    check_point_filename = 'model_checkpoint.pt'

    # training
    print('Training the network...')
    max_pc = pc
    for epoch in range(n_epochs):
        t0 = time.perf_counter()
        if loss == 'boundary':
            loss_fn.alpha = alpha
            alpha -= 0.005 if alpha > 0.75 else alpha

        print('*------------------------------------------------------*')
        train_loss, train_acc = run_one_epoch(net, train_dataloader, train_distmap, optimizer, loss_fn, train=True,
                                              device=device)
        val_loss, val_acc = run_one_epoch(net, val_dataloader, val_distmap, optimizer, loss_fn, device=device)

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), check_point_filename)
        else:
            pc -= 1

        if pc <= 0:
            if early_stop:  # early stopping
                break
            else:  # halved lr every time patience limit is reached
                pc = max_pc
                lr /= 2
                optimizer = torch.optim.Adam(net.parameters(), lr=lr, amsgrad=True)

        delta_t = time.perf_counter() - t0
        print("Epoch %i took %.2fs. Train loss: %.4f acc: %.4f. Val loss: %.4f acc: %.4f. Patience left: %i" %
              (epoch + 1, delta_t, train_loss, train_acc, val_loss, val_acc, pc))

    return train_accs, train_losses, val_accs, val_losses


def joint_pred(data_path, model_path, mask_option, sigma=None, display=False):
    """Joint prediction with nuclei prediction model, watershed over ecad images"""
    pred_results = []
    print('Loading datasets...')
    print('- Test set {0}:'.format(data_path.split('/')[-2]))
    dapi_dataset, _ = load_data(data_path, 'dapi', 'dapi', mask_option='multi', sigma=sigma, enhance=True)
    ecad_dataset, _ = load_data(data_path, 'ecad', 'ecad', mask_option='multi', sigma=sigma)
    dapi_dataloader = data.DataLoader(dapi_dataset, batch_size=1)
    ecad_dataloader = data.DataLoader(ecad_dataset, batch_size=1)
    
    # Initialize network & training, transfer to GPU is available
    c_out = 1 if mask_option == 'binary 'else 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # debug
    nuclei_net = Unet(1, c_out)
    nuclei_net.load_state_dict(torch.load(model_path))
    nuclei_net.to(device)
    
    print('Predicting test results...')
    bar = ChargingBar('Test', max=len(dapi_dataloader), suffix='%(percent)d%%')
    for dapi, ecad in zip(dapi_dataloader, ecad_dataloader):
        bar.next()
        dapi_img, ecad_img = dapi[0].float().to(device), ecad[0].float().to(device)
        joint_pred = Postprocessor(ecad_img, nuclei_net(dapi_img)).out
        pred_results.append(joint_pred)

        if display:
            plt.figure(figsize=(20, 10))
            plt.subplot(1,2,1)
            plt.imshow(dapi_img.detach().cpu().squeeze().numpy(), cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(joint_pred, cmap='gray')
            plt.show()
            plt.close()
            
    bar.finish()

    return pred_results


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


def save_history(train_accs, train_losses, val_accs, val_losses, out_path=None):
    """Save accuracies & losses"""
    if out_path == None:
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


def save_test_pred(results, data_path, out_path=None):
    """Save postprocessed test predictions"""
    print('Saving predicted images...')
    prefix = data_path.split('/')[-2]
    if out_path is None:
        out_path = '../predictions/'
    os.makedirs(out_path, exist_ok=True)

    for i, res in enumerate(results):
        fname = prefix + '_' + str(i) + '.png'
        cv2.imwrite(out_path + fname, res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet training options',
            formatter_class=RawTextHelpFormatter)
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', dest='root_path', type=str, required=True, action='store',
                        help='Root directory of input image datasets for training/testing')
    required.add_argument('--option', dest='option', type=str, required=True, action='store',
                        help='Training option: (1). binary, (2). multi, (3). dwt')

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-b', dest='batch_size', type=int, default=4, action='store',
                        help='Batch size')
    optional.add_argument('-l', dest='loss', type=str, default='bce', action='store',
                        help='Loss function\n  Options: (1). bce; (2). jaccard; (3).dice; (4).boundary'),
    optional.add_argument('-n', dest='n_epochs',  type=int, default=150, action='store',
                        help='Total number of epoches for training')
    optional.add_argument('-m', dest='model', type=str, default='./model_checkpoint.pt', action='store',
                        help='Saved model file')
    optional.add_argument('-r', dest='lr', type=float, default=0.01, action='store',
                        help='Learning rate')
    optional.add_argument('-p', dest='patience_counter', type=int, default=30, action='store',
                        help='Patience counter for early-stopping or lr-tuning')
    optional.add_argument('--test', dest='test', action='store_true',
                        help='Whether perform prediction & postprocessing on the test set')
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
    option = args.option
    augment = args.augment
    n_epochs = args.n_epochs
    lr = args.lr
    batch_size = args.batch_size
    patience_counter = args.patience_counter
    early_stop = args.early_stop
    loss = args.loss
    region_option = args.region_option
    dist = None  # weighted distmap indicator
    alpha = None  # parameter for boundary loss

    # data augmentation on training
    if augment:
        print('Performing data augmentation...')
        augmentation(root_path, mode='train')
        augmentation(root_path, mode='val')
        augmentation(root_path, mode='test')
        exit()

    if loss == 'bce':
        dist = 'dist' if option == 'binary' else 'saw'
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

    # todo: (1). finish DWT implementation (2). Incorporate predictions with topo-loss results
    if not args.test:
        train_accs, train_losses, val_accs, val_losses = train(root_path=root_path,
                                                               bs=batch_size,
                                                               lr=lr,
                                                               pc=patience_counter,
                                                               mask_option=option,
                                                               dist=dist,
                                                               sigma=sigma,
                                                               loss_fn=loss_fn,
                                                               alpha=alpha)
        save_history(train_accs, train_losses, val_accs, val_losses)
    else:
        pred_results = joint_pred(data_path=root_path, model_path=args.model, mask_option=option)
        save_test_pred(results=pred_results, data_path=root_path)

