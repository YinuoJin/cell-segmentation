import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from progress.bar import ChargingBar
from torch.utils import data
from dataset import load_data, augmentation, calc_weight
from model import Unet, VGG19
from utils import vgg_topo_loss


def run_one_epoch(dataloader, model, vgg_model, optimizer, train=False, device=None, weight=1):
    torch.set_grad_enabled(train)
    model.train() if train else model.eval()
    
    losses = []
    accuracies = []
    
    if train:
        bar = ChargingBar('Training', max=len(dataloader), suffix='%(percent)d%%')
    else:
        bar = ChargingBar('Validation', max=len(dataloader), suffix='%(percent)d%%')
    
    for i, (x, y) in enumerate(dataloader):
        bar.next()
        x, y = x.float(), y.float()  # Type conversion (avoid bugs of DOUBLE <==> FLOAT
        x, y = x.to(device), y.to(device)
        output = model(x)
        
        # Pipe both ground truth masks & prediction into VGG model, extract feature maps for loss function calculation
        loss = vgg_topo_loss(vgg_model, output, y, weight=weight)
        #loss = F.binary_cross_entropy_with_logits(output, y, weight=torch.Tensor([weight]))
        
        if train:  # backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        losses.append(loss.detach().numpy())
        accuracy = torch.mean(((output > 0.5) == (y > 0.5)).float())
        accuracies.append(accuracy.detach().numpy())
        
    bar.finish()
    
    return np.mean(losses), np.mean(accuracies)


if __name__ == '__main__':
    root_path = '../datasets/multi_cell_custom_data_without_shapes/'
    
    # data augmentation on training & validation sets
    """
    print('Performing data augmentation...')
    augmentation(root_path)
    augmentation(root_path, mode='val')
    """
    
    # load dataset
    print('Loading datasets...')
    print('- Training set:')
    train_dataset, _, train_mat_mask = load_data(root_path, 'train_frames_aug', 'train_masks_aug', return_matrix=True)
    print('- Validation set:')
    val_dataset = load_data(root_path, 'val_frames_aug', 'val_masks_aug')
    # test_dataset = load_data(root_path, 'test_frames', 'test_masks')
    
    weight = calc_weight(train_mat_mask)
    
    train_dataloader = data.DataLoader(train_dataset, batch_size=1)
    val_dataloader = data.DataLoader(val_dataset, batch_size=1)
    # test_dataloader = data.DataLoader(test_dataset, batch_size=1)
    
    # Initialize network & training, transfer to GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Unet(3)
    vgg_model = VGG19()
    net.to(device)
    vgg_model.to(device)
    
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    best_val_loss = np.inf
    check_point_filename = 'model_checkpoint.pt'
    
    patience = 10
    patience_counter = 10
    optimizer = torch.optim.Adam(net.parameters())
    
    # training
    print('Training the network...')
    for epoch in range(50):
        t0 = time.perf_counter()
        print('|------------------------------------------------------|')
        train_loss, train_acc = run_one_epoch(train_dataloader, net, vgg_model, optimizer, train=True, device=device)
        val_loss, val_acc = run_one_epoch(val_dataloader, net, vgg_model, optimizer, device=device)
        print('|------------------------------------------------------|')
    
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
            break
    
        delta_t = time.perf_counter() - t0
        print("Epoch %i took %.2fs. Train loss: %.4f acc: %.4f. Val loss: %.4f acc: %.4f. Patience left: %i" %
              (epoch + 1, delta_t, train_loss, train_acc, val_loss, val_acc, patience_counter))
