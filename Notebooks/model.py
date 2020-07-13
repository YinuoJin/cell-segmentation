import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg19


class Unet(nn.Module):
    """
    --\          ->          /--
       --\       ->       /--
          --\    ->    /--
             --\ -> /--
                 --

    - : Conv
    \ : Maxpool
    / : Upsample (ConvTransposed)
    ->: Concatenate
    """
    
    def __init__(self, n_channels, dropout=0.1):
        super(Unet, self).__init__()
        self.contract_net = Contraction(n_channels, dropout)
        self.double_conv_net = DoubleConv2d(512, 1024, dropout)
        self.expand_net = Expansion(dropout)
        self.out_net = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x, conv_output = self.contract_net(x)
        x = self.double_conv_net(x)
        x = self.expand_net(x, conv_output)
        x = self.out_net(x)
        
        return x


class Contraction(nn.Module):
    """
    Contracting path of Unet (Encoder)
    --\
       --\
          --\
             --\
    """
    
    def __init__(self, in_channel, dropout=0.1):
        super(Contraction, self).__init__()
        self.dconv_net1 = DoubleConv2d(in_channel, 64, dropout)
        self.dconv_net2 = DoubleConv2d(64, 128, dropout)
        self.dconv_net3 = DoubleConv2d(128, 256, dropout)
        self.dconv_net4 = DoubleConv2d(256, 512, dropout)
        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x):
        conv1 = self.dconv_net1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_net2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_net3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_net4(x)
        x = self.maxpool(conv4)
        
        return x, (conv1, conv2, conv3, conv4)


class Expansion(nn.Module):
    """
    Expanding path of Unet (Decoder)
             /--
          /--
       /--
    /--
    """
    
    def __init__(self, dropout=0.1):
        super(Expansion, self).__init__()
        self.upsample1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.uconv_net1 = DoubleConv2d(1024, 512, dropout)
        
        self.upsample2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.uconv_net2 = DoubleConv2d(512, 256, dropout)
        
        self.upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.uconv_net3 = DoubleConv2d(256, 128, dropout)
        
        self.upsample4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.uconv_net4 = DoubleConv2d(128, 64, dropout)
    
    def forward(self, x, contract_conv_output):
        dconv1, dconv2, dconv3, dconv4 = contract_conv_output
        
        x = self.upsample1(x)
        x = self.concatenate(x, dconv4)
        x = self.uconv_net1(x)
        
        x = self.upsample2(x)
        x = self.concatenate(x, dconv3)
        x = self.uconv_net2(x)
        
        x = self.upsample3(x)
        x = self.concatenate(x, dconv2)
        x = self.uconv_net3(x)
        
        x = self.upsample4(x)
        x = self.concatenate(x, dconv1)
        x = self.uconv_net4(x)
        
        return x
    
    @staticmethod
    def concatenate(a, b, axis=1):
        concat_size = np.min([a.shape[2], b.shape[2]])
        if (a.shape[2] > concat_size):
            a, b = b, a
        
        midpoint = int(b.shape[2] // 2)  # [-----|-----]
        span = int(concat_size // 2)  # [ <---|---> ] trim the edge of the matrix, reshape for concatenation
        b = b[:, :, midpoint - span:midpoint + span, midpoint - span:midpoint + span]
        
        return torch.cat((a, b), axis=axis)


class DoubleConv2d(nn.Module):
    
    def __init__(self, in_channel, out_channel, dropout=0.1):
        super(DoubleConv2d, self).__init__()
        conv_layers = [
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(dropout),
            nn.ELU(inplace=True),
            
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(dropout),
            nn.ELU(inplace=True)
        ]
        
        self.conv_net = nn.Sequential(*conv_layers)
    
    def forward(self, x):
        return self.conv_net(x)


class VGG19(torch.nn.Module):
    """VGG19 pretrained model for the implementation of  topo-awared loss function"""
    # reference: https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113/2
    def __init__(self):
        super(VGG19, self).__init__()
        self.features = vgg19(pretrained=True).features.eval()
    
    def forward(self, x, inter_layer_indices=[2, 7, 16], last_layer_idx=17):
        assert max(inter_layer_indices) < last_layer_idx, 'Layer index out of bound!'
        res = []
        for i, layer in enumerate(self.features[:last_layer_idx]):
            x = layer(x)
            if i in inter_layer_indices:
                res.append(x)
        
        return res
