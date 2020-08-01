import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from progress.bar import ChargingBar
from torch.utils import data
from scipy.spatial.distance import directed_hausdorff
from dataset import load_data, augmentation, calc_weight
from model import Unet
from utils import IoULoss, SoftDiceLoss, SurfaceLoss, DistWeightBCELoss


def run_one_epoch(model, dataloader, distmap, optimizer, loss_fn, train=False, device=None):
    torch.set_grad_enabled(train)
    model.train() if train else model.eval()
    losses = []
    accuracies = [] 

    if train:
        bar = ChargingBar('Train', max=len(dataloader), suffix='%(percent)d%%')
    else:
        bar = ChargingBar('Valid', max=len(dataloader), suffix='%(percent)d%%')
    
    for i, (dp, theta) in enumerate(zip(dataloader, distmap)):
        bar.next()
        x, y = dp
        x, y, theta = x.float(), y.float(), theta.float() # Type conversion (avoid bugs of DOUBLE <==> FLOAT)
        x, y, theta = x.to(device), y.to(device), theta.to(device)
        output = model(x)
        
        loss = loss_fn(y, output, theta)
        
        if train:  # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses.append(loss.detach().cpu().numpy())

        # tmp: assign hausdorff distance as "accuracy
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
    root_path = '../datasets/multi_cell_custom_data_without_shapes/'
    
    # data augmentation on training & validation sets
    # print('Performing data augmentation...')
    # augmentation(root_path)
    # augmentation(root_path, mode='val')
    
    # load dataset
    print('Loading datasets...')
    print('- Training set:')
    train_dataset, train_distmap = load_data(root_path, 'train_frames_aug', 'train_masks_aug', return_dist='boundary')
    print('- Validation set:')
    val_dataset, val_distmap = load_data(root_path, 'val_frames_aug', 'val_masks_aug', return_dist='boundary')
    # test_dataset = load_data(root_path, 'test_frames', 'test_masks')
    
    train_dataloader = data.DataLoader(train_dataset, batch_size=8)
    train_distmap = data.DataLoader(train_distmap, batch_size=8)
    val_dataloader = data.DataLoader(val_dataset, batch_size=8)
    val_distmap = data.DataLoader(val_distmap, batch_size=8)
    # test_dataloader = data.DataLoader(test_dataset, batch_size=1)
    
    # Initialize network & training, transfer to GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Unet(1)
    net.to(device)
    
    train_accs, train_losses, val_accs, val_losses = [], [], [], []
    best_val_loss = np.inf
    check_point_filename = 'model_checkpoint.pt'
    
    patience_counter = 20
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, amsgrad=True)
    
    # todo: try with multiple losses
    # if boundary loss, alpha varies based on epoch
    alpha = 1.0
    # loss_fn = IoULoss()
    # loss_fn = SoftDiceLoss()
    # loss_fn = DistWeightBCELoss()
    
    # training
    print('Training the network...')
    for epoch in range(200):
        t0 = time.perf_counter()
        loss_fn = SurfaceLoss(alpha=alpha, dice=False)
        alpha -= 0.005 if alpha > 0.25 else alpha
        
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
            patience_counter = 20
            lr /= 2
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, amsgrad=True)
        
        delta_t = time.perf_counter() - t0
        print("Epoch %i took %.2fs. Train loss: %.4f acc: %.4f. Val loss: %.4f acc: %.4f. Patience left: %i" %
              (epoch + 1, delta_t, train_loss, train_acc, val_loss, val_acc, patience_counter))

    # Save accuracies & losses
    out_path = '../results/'
    train_accs = np.array(train_accs)
    train_losses = np.array(train_losses)
    val_accs = np.array(val_accs)
    val_losses = np.array(val_losses)

    np.savetxt(out_path + 'acc_train.txt', train_accs)
    np.savetxt(out_path + 'loss_train.txt', train_losses)
    np.savetxt(out_path + 'acc_val.txt', val_accs)
    np.savetxt(out_path + 'loss_val.txt', val_losses)
