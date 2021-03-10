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
from model import Unet, ConvRecUnet, ResUnet, FPN
from postprocessing import Postprocessor, class_assignment
from utils import hausdorff, calc_accuracy_score, calc_f1_score
from utils import IoULoss, SoftDiceLoss, SurfaceLoss, ShapeBCELoss


class Arguments:
    """
    Wrapper for arguments for "train" function
    """
    def __init__(self, **kwargs):
        self.root_path = kwargs['root_path']
        self.out_path = kwargs['out_path']
        self.model_path = kwargs['model_path']
        self.image_type = kwargs['image_type']
        self.early_stop = kwargs['early_stop']
        self.net = kwargs['net']
        self.n_epochs = kwargs['n_epochs']
        self.bs = kwargs['batch_size']
        self.lr = kwargs['learning_rate']
        self.pc = kwargs['max_patience']
        self.mask_option = kwargs['mask_option']
        self.dist = kwargs['dist']
        self.sigma = kwargs['sigma']
        self.loss_name = kwargs['loss_name']
        self.loss_fn = kwargs['loss_function']
        self.alpha = kwargs['alpha']


def train(args):
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    net = args.net
    print('Loading datasets...')
    print('- Training set:')
    train_dataloader, train_distmap = load_data(root_path=args.root_path,
                                                image_type=args.image_type,
                                                frame='train_frames',
                                                mask='train_masks',
                                                mask_option=args.mask_option,
                                                sigma=args.sigma,
                                                batch_size=args.bs,
                                                return_dist=args.dist)
    print('- Validation set:')
    val_dataloader, val_distmap = load_data(root_path=args.root_path,
                                            image_type=args.image_type,
                                            frame='val_frames',
                                            mask='val_masks',
                                            mask_option=args.mask_option,
                                            sigma=args.sigma,
                                            batch_size=args.bs,
                                            return_dist=args.dist)

    # Initialize network & training, transfer to GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    train_accs, train_losses, val_accs, val_losses, train_dists, val_dists = [], [], [], [], [], []
    best_val_loss = np.inf
    best_val_dist = np.inf
    check_point_filename = 'model_checkpoint.pt'

    # training
    print('Training the network...')
    max_pc = args.pc
    for epoch in range(args.n_epochs):
        t0 = time.perf_counter()
        if args.loss_name == 'boundary':
            loss_fn.alpha = alpha
            alpha -= 0.005 if alpha > 0.75 else alpha

        print('*------------------------------------------------------*')
        train_loss, train_acc, train_dist = run_one_epoch(net, train_dataloader, train_distmap, optimizer, args.loss_fn, args.loss_name,
                                              option=args.mask_option, train=True, device=device)
        val_loss, val_acc, val_dist = run_one_epoch(net, val_dataloader, val_distmap, optimizer, args.loss_fn, args.loss_name,
                                          option=args.mask_option, device=device)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        train_dists.append(train_dist)
        val_dists.append(val_dist)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), os.path.join(args.out_path, check_point_filename))
        else:
            args.pc -= 1

        if args.pc <= 0:
            if args.early_stop:  # early stopping
                break
            else:  # halved lr every time patience limit is reached
                args.pc = max_pc
                args.lr /= 2
                optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)

        delta_t = time.perf_counter() - t0
        print("Epoch %i took %.2fs. Train loss: %.4f acc: %.4f. dist: %.4f. Val loss: %.4f acc: %.4f. dist: %.4f. Patience left: %i" %
              (epoch + 1, delta_t, train_loss, train_acc, train_dist, val_loss, val_acc, val_dist, args.pc))

    return train_accs, train_losses, train_dists, val_accs, val_losses, val_dists


def pred(data_path, model_path, net, mask_option, sigma=1, display=False, return_binary=True, return_contour=True):
    """General prediction with watershed postprocessing"""

    def tmp_class_assignment(mask, t1=0.5, t2=0.5):
        """
        Assign class labels from segmentation predictions

        Parameters
        ----------
        t1 : float
            threshold 1 - cutoff for class 0 & 1 assignment (background & cell foreground)

        t2 : float
            threshold 2 - cutoff for class 2 asssignment (attaching border)
        """
        raw_output =  np.argmax(mask, axis=0)
        output = np.zeros_like(mask)
        output[0][raw_output == 0] = 1
        output[1][raw_output == 1] = 1
        output[2][raw_output == 2] = 1

        return output

    pred_results = []
    print('Loading datasets...')

    test_dataloader, _ = load_data(data_path, None, 'val_frames', 'val_masks', mask_option=mask_option, sigma=sigma, batch_size=1)

    # Initialize network & training, transfer to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(model_path))
    net.to(device)

    print('Predicting test results...')
    bar = ChargingBar('Test', max=len(test_dataloader), suffix='%(percent)d%%')
    for (x, y) in test_dataloader:
        bar.next()
        x, y = x.float().to(device), y.float().to(device)
        y_pred = net(x)
        if y_pred.shape[1] == 3: # binarize
            y_pred_copy = y_pred.clone()
            y_pred = y_pred.detach().cpu().squeeze().numpy()
            y_pred_binary = tmp_class_assignment(y_pred)
            pred_results.append(y_pred_binary)
            y_pred_ws = Postprocessor(y_pred_copy, return_contour=return_contour, return_binary=return_binary).out
        else:
            y_pred_binary = y_pred.detach().cpu().squeeze().numpy()
            y_pred_ws = Postprocessor(y_pred, return_contour=return_contour, return_binary=return_binary).out
            pred_results.append(y_pred_ws)

        if display:
            plt.figure(figsize=(30, 10))
            plt.subplot(1,3,1)
            plt.imshow(x.detach().cpu().squeeze().numpy(), cmap='gray')
            plt.subplot(1,3,2)
            if y_pred_binary.ndim == 3:
                plt.imshow(y_pred_binary.transpose((1,2,0)))
            else:
                plt.imshow(y_pred_binary, cmap='gray')
            plt.subplot(1,3,3)
            plt.imshow(y_pred_ws, cmap='gray')
            plt.show()
            plt.close()

    bar.finish()

    return pred_results


def run_one_epoch(model, dataloader, distmap, optimizer, loss_fn, loss_name, option=None, train=False, device=None):
    """Single epoch training/validating"""
    torch.set_grad_enabled(train)
    model.train() if train else model.eval()
    losses = []
    accuracies = []
    distances = []  # Hausdorff distance

    if train:
        bar = ChargingBar('Train', max=len(dataloader), suffix='%(percent)d%%')
    else:
        bar = ChargingBar('Valid', max=len(dataloader), suffix='%(percent)d%%')

    if loss_name == 'dice' or loss_name == 'jaccard': # loss functions without distmap
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

            if option == 'binary':
                accuracy =  torch.mean(((output > 0.5) == (y > 0.5)).float()).detach().cpu().numpy()
            elif option == 'multi' or 'fpn':
                accuracy = calc_f1_score(y, output)
            dist = hausdorff(y, output)

            accuracies.append(accuracy)
            distances.append(dist)

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

            if option == 'binary':
                accuracy = torch.mean(((output > 0.5) == (y > 0.5)).float()).detach().cpu().numpy()
                distance = hausdorff(y, output)
            elif option == 'multi' or 'fpn':
                # debug: measure hausdorff distance as accuracy
                accuracy = calc_f1_score(y, output)
                distance = hausdorff(y, output)

            accuracies.append(accuracy)
            distances.append(distance)

            del x, y, dist
            torch.cuda.empty_cache()


    bar.finish()

    return np.mean(losses), np.mean(accuracies), np.mean(distances)


def save_history(training_history, out_path=None):
    """Save accuracies & losses"""
    train_accs, train_losses, train_dists, val_accs, val_losses, val_dists = training_history
    if out_path == None:
        out_path = '../results/'
    os.makedirs(out_path, exist_ok=True)
    train_accs, train_losses, train_dists = np.array(train_accs), np.array(train_losses), np.array(train_dists)
    val_accs, val_losses, val_dists = np.array(val_accs), np.array(val_losses), np.array(val_dists)

    np.savetxt(out_path + 'acc_train.txt', train_accs)
    np.savetxt(out_path + 'loss_train.txt', train_losses)
    np.savetxt(out_path + 'dist_train.txt', train_dists)
    np.savetxt(out_path + 'acc_val.txt', val_accs)
    np.savetxt(out_path + 'loss_val.txt', val_losses)
    np.savetxt(out_path + 'dist_val.txt', val_dists)


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
                        help='Training option: (1). binary, (2). multi')
    required.add_argument('--image-type', dest='image_type', type=str, default='nuclei', required=True, action='store',
                         help='Image type: (1). nuclei, (2). membrane')
    
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-o', dest='out_path', type=str, default='./', action='store',
                        help='Directory to output file')
    optional.add_argument('-a', dest='net_option', type=str, default='unet', action='store',
                        help='Model Architecture\n Options: (1).unet; (2).resnet; (3).convnet; (4). fpn')
    optional.add_argument('-d', dest='dist', type=str, default=None, action='store',
                        help='Distance function for weighted loss\n Options: (1).dist; (2).saw; (3).class; (4).boundary')
    optional.add_argument('-b', dest='batch_size', type=int, default=4, action='store',
                        help='Batch size')
    optional.add_argument('-l', dest='loss', type=str, default='bce', action='store',
                        help='Loss function\n  Options: (1).bce; (2).jaccard; (3).dice; (4).boundary'),
    optional.add_argument('-n', dest='n_epochs',  type=int, default=50, action='store',
                        help='Total number of epoches for training')
    optional.add_argument('-m', dest='model_path', type=str, default='./model_checkpoint.pt', action='store',
                        help='Saved model file')
    optional.add_argument('-r', dest='lr', type=float, default=0.01, action='store',
                        help='Learning rate')
    optional.add_argument('-p', dest='patience_counter', type=int, default=30, action='store',
                        help='Patience counter for early-stopping or lr-tuning')
    optional.add_argument('--test', dest='test', action='store_true',
                        help='Whether to perform prediction & postprocessing on the test set')
    optional.add_argument('--joint-pred', dest='joint_pred', action='store_true',
                        help='Whether to perform joint prediction & postprocessing on the test set (with both nuclei & membrane marker channels)')
    optional.add_argument('--augment', dest='augment', action='store_true',
                        help='Whether to perform data augmentation in the current run')
    optional.add_argument('--early-stop', dest='early_stop', action='store_true',
                        help='Whether to perform early-stopping; If False, lr is halved when reaching each patience')
    optional.add_argument('--region-option', dest='region_option', action='store_true',
                        help='Whether to use dice loss as the Region-based loss for boundary loss; If False, jaccard loss is used instead')

    parser.set_defaults(feature=True)
    args = parser.parse_args()

    # Parameter initialization
    alpha = None  # parameter for boundary loss
    sigma = 3  # parameter for gaussian blur in weighted distmap calculation
    
    # data augmentation on training
    if args.augment:
        print('Performing data augmentation...')
        augmentation(args.root_path, mode='train')
        augmentation(args.root_path, mode='val')
        augmentation(args.root_path, mode='test')
        exit()

    # Configure loss function type and distance option
    if args.loss == 'bce' and args.option == 'binary':
        loss_fn = ShapeBCELoss()
    elif args.loss == 'bce' and args.option == 'multi':
        # loss_fn = ShapeBCELoss() if args.option == 'multi' else FPNLoss()
        loss_fn = ShapeBCELoss()
    elif args.loss == 'jaccard':
        loss_fn = IoULoss()
    elif args.loss == 'dice':
        loss_fn = SoftDiceLoss()
    elif args.loss == 'boundary':
        alpha = 1.0  # a: (a * Region-based loss + (1-a) * boundary loss)
        loss_fn = SurfaceLoss(alpha=alpha, dice=args.region_option)
    else:
        raise NotImplementedError('Loss function {0} not recognized'.format(loss))

    # Configure network architecture option
    c_in = 1
    if args.option == 'binary':
        c_out = 1
    elif args.option == 'multi':
        c_out = 3
    else:
        raise NotImplementedError('Invalid mask option {0}, available options: binary; multi; fpn').format(args.option)

    if args.net_option == 'unet':
        net = Unet(c_in, c_out)
    elif args.net_option == 'resnet':
        print('Using ResNet instead of normal U-net...')
        net = ResUnet(c_in, c_out)
    elif args.net_option == 'lstm':
        print('Using ConvLSTMNet instead of normal U-net...')
        net = ConvRecUnet(c_in, c_out)
    elif args.net_option == 'fpn':
        print('Using FPN instead of normal U-net...')
        net = FPN(c_in, c_out, c_1=128)
    else:
        raise NotImplementedError('Invalid model architecture option {0}, available options: unet, resnet, lstm, fpn').format(args.net_option)

    train_args = Arguments(
        root_path=args.root_path,
        out_path=args.out_path,
        model_path=args.model_path,
        image_type=args.image_type,
        early_stop=args.early_stop,
        net=net,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_patience=args.patience_counter,
        mask_option=args.option,
        dist=args.dist,
        sigma=sigma,
        loss_name=args.loss,
        loss_function=loss_fn,
        alpha=alpha
    )

    if not args.test:
        training_history = train(train_args)
        save_history(training_history, out_path=args.out_path)
    else:
        if args.joint_pred:
            pred_results = joint_pred(data_path=args.root_path, model_path=args.model_path, net=net, mask_option=args.option)
        else:
            pred_results = pred(data_path=args.root_path, model_path=args.model_path, net=net, mask_option=args.option)
        # save_test_pred(results=pred_results, data_path=args.root_path)
