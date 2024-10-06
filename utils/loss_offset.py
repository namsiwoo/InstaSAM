import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, distance_transform_edt

def norm(img):
    if torch.max(img) != 0:
        # print(torch.max(img))
        return img/torch.max(img)
    else:
        return img

class offset_Loss(nn.Module):

    def __init__(self, W, H):
        super(offset_Loss, self).__init__()
        self.l1 = torch.nn.SmoothL1Loss(reduction='none')

        self.x_map = torch.zeros(W, H)
        for i in range(W):
            self.x_map[:, i] = i
        self.y_map = torch.zeros(W, H)
        for i in range(H):
            self.y_map[i, :] = i

    def forward(self, pred, mask, ignored_map=None):
        B, W, H = mask.shape
        gt = torch.zeros((B, 2, W, H), dtype=torch.float32).to(pred.device.index)
        self.x_map = self.x_map.to(pred.device.index)
        self.y_map = self.y_map.to(pred.device.index)

        if ignored_map is None:
            ignored_map = torch.ones_like(mask)

        for b in range(B):
            index_list = torch.unique(mask[b])
            for index in index_list[1:]:
                nuclei = (mask[b] == index)
                y, x = torch.nonzero(nuclei, as_tuple=True)

                y, x = y.float().mean(), x.float().mean()
                gt[b, 0] = gt[b, 0] + norm((self.x_map-x)*nuclei)
                gt[b, 1] = gt[b, 1] + norm((self.y_map-y)*nuclei)

        fg_loss = (self.l1(pred, gt.float()) * (mask != 0) / ((mask != 0)+1e-7).sum()) * ignored_map
        bg_loss = (self.l1(pred, gt.float()) * (mask == 0) / ((mask == 0)+1e-7).sum()) * ignored_map
        offset_loss = torch.sum((fg_loss+bg_loss))
        return offset_loss, gt