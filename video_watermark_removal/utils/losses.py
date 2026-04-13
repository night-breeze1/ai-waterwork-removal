
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15, 22]):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = layers
        self.model = nn.Sequential()
        
        for i, layer in enumerate(vgg):
            self.model.add_module(str(i), layer)
            if i == max(layers):
                break
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
    def forward(self, x, y):
        loss = 0.0
        x_feat = x
        y_feat = y
        
        for i, layer in enumerate(self.model):
            x_feat = layer(x_feat)
            y_feat = layer(y_feat)
            
            if i in self.layers:
                loss += F.mse_loss(x_feat, y_feat)
        
        return loss


class TVLoss(nn.Module):
    def __init__(self, weight=1e-4):
        super(TVLoss, self).__init__()
        self.weight = weight
    
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class AdversarialLoss(nn.Module):
    def __init__(self, gan_type='lsgan'):
        super(AdversarialLoss, self).__init__()
        self.gan_type = gan_type
        if gan_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif gan_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target_is_real):
        if self.gan_type == 'lsgan':
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        else:
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.criterion(pred, target)
