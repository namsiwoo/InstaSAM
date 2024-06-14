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
        # y_gt = torch.zeros((B, W, H), dtype=torch.float32).to(self.device)

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

                # y, x = torch.round(y.float().mean()), torch.round(x.float().mean())
                # y, x = int(y), int(x)
                # dis_transform = distance_transform_edt(nuclei.detach().cpu().numpy())
                # dis_transform = torch.from_numpy(dis_transform).to(pred.device.index)
                # dis_transform = torch.max(dis_transform) - dis_transform
                #
                # dist_x = torch.sqrt(abs(dis_transform**2 - (self.x_map-x)**2))
                # dist_x[:, :x] = dist_x[:, :x]*(-1)
                #
                # dist_y = torch.sqrt(abs(dis_transform**2 - (self.y_map-y)**2))
                # dist_y[y:, :] = dist_y[y:, :]*(-1)
                #
                # gt[b, 0] = gt[b, 0] + norm(dist_x*nuclei)
                # gt[b, 1] = gt[b, 1] + norm(dist_y*nuclei)


                # gt[b, 0, nuclei == True] = norm((self.x_map-x)*nuclei)
                # gt[b, 1, nuclei ==True] = norm((self.y_map-y)*nuclei)
                # if torch.max((self.x_map-x)*nuclei) == 0:
                #     aaa = mask[b].detach().cpu().numpy()
                #     print(torch.sum(nuclei))
                #     from PIL import Image
                #     aaa = Image.fromarray(aaa.astype(np.uint8)).convert('L')
                #     aaa.save('/media/NAS/nas_187/siwoo/2023/result/sam_offset_normoffset/x_{}_{}_mask.png')
                #
                #     plt.clf()
                #     plt.subplot(1, 3, 1)
                #     plt.imshow(mask[b].detach().cpu())
                #
                #     plt.subplot(1, 3, 2)
                #     plt.imshow(nuclei.detach().cpu())
                #
                #     plt.subplot(1, 3, 3)
                #     plt.title(torch.sum(nuclei).detach().cpu().numpy())
                #     plt.imshow(((self.x_map-x)*nuclei).detach().cpu().numpy())
                #     plt.colorbar()
                #     plt.savefig('/media/NAS/nas_187/siwoo/2023/result/sam_offset_normoffset/x_{}_{}_.png'.format(str(b), str(index)))

        fg_loss = (self.l1(pred, gt.float()) * (mask != 0) / ((mask != 0)+1e-7).sum()) * ignored_map
        bg_loss = (self.l1(pred, gt.float()) * (mask == 0) / ((mask == 0)+1e-7).sum()) * ignored_map
        offset_loss = torch.sum((fg_loss+bg_loss))
        return offset_loss, gt

class offset_Loss_sonnet(nn.Module):

    def __init__(self, W, H):
        super(offset_Loss_sonnet, self).__init__()
        self.l1 = torch.nn.SmoothL1Loss(reduction='none')
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
        # self.ce = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.x_map = torch.zeros(W, H)
        for i in range(W):
            self.x_map[:, i] = i
        self.y_map = torch.zeros(W, H)
        for i in range(H):
            self.y_map[i, :] = i

    def forward(self, pred, mask, ignored_map=None):
        B, W, H = mask.shape
        gt = torch.zeros((B, W, H), dtype=torch.float32).to(pred.device.index)

        self.DDD_list = [0.83, 0.68, 0.54, 0.41, 0.29, 0.18, 0.09, 0.0]
        # self.DDD_list = [0.83, 0.54, 0.29, 0.0]


        gt_sonnet = torch.zeros((B, len(self.DDD_list), W, H), dtype=torch.float32).to(pred.device.index) # each ord
        self.x_map = self.x_map.to(pred.device.index)
        self.y_map = self.y_map.to(pred.device.index)

        if ignored_map is None:
            ignored_map = torch.ones_like(pred)
            # plt.clf()
            # plt.subplot(1, 2, 1)
            # plt.imshow(ignored_map.detach().cpu()[0])
            # plt.subplot(1, 2, 2)
            # plt.imshow(mask.detach().cpu()[0])
            # plt.savefig('/media/NAS/nas_187/siwoo/2023/result/sam_ssl_ord/ignored.png')
            # print(aa)

        for b in range(B):
            index_list = torch.unique(mask[b])
            for index in index_list[1:]:
                nuclei = (mask[b] == index)

                #original
                # y, x = torch.nonzero(nuclei, as_tuple=True)
                # # y, x = torch.round(y.float().mean()), torch.round(x.float().mean())
                # y, x = y.float().mean(), x.float().mean()
                #
                # gt[b] = gt[b] + ((1- norm((((self.x_map-x)**2 + (self.y_map-y)**2)**(1/2))*nuclei))*nuclei)>0.5
                # for i in range(len(self.DDD_list)):
                #     if i == 0:
                #         binary = norm((((self.x_map - x) ** 2 + (self.y_map - y) ** 2) ** (1 / 2)) * nuclei) >= self.DDD_list[i]
                #     else:
                #         binary = (((norm((((self.x_map - x) ** 2 + (self.y_map - y) ** 2) ** (1 / 2)) * nuclei) >= self.DDD_list[i]) *
                #                   (norm((((self.x_map - x) ** 2 + (self.y_map - y) ** 2) ** (1 / 2)) * nuclei) < self.DDD_list[i-1]))) * nuclei
                #     gt_sonnet[b, i] = gt_sonnet[b, i] + binary

                #new offset
                dis_transform = distance_transform_edt(nuclei.detach().cpu().numpy())
                dis_transform = torch.from_numpy(dis_transform).to(pred.device.index)
                dis_transform = norm(dis_transform)
                dis_transform[dis_transform < 0.5] = 0
                dis_transform[dis_transform > 0.7] = 1

                gt[b] = gt[b] + dis_transform

                for i in range(len(self.DDD_list)):
                    if i == 0:
                        binary = dis_transform >= self.DDD_list[i]
                    else:
                        binary = (((dis_transform >= self.DDD_list[i]) *
                                  (dis_transform < self.DDD_list[i-1]))) * nuclei
                    gt_sonnet[b, i] = gt_sonnet[b, i] + binary



                # gt[b, 0] += norm((self.x_map-x)*nuclei)
                # gt[b, 1] += norm((self.y_map-y)*nuclei)
                # if torch.max((self.x_map-x)*nuclei) == 0:
                #     aaa = mask[b].detach().cpu().numpy()
                #     print(torch.sum(nuclei))
                #     from PIL import Image
                #     aaa = Image.fromarray(aaa.astype(np.uint8)).convert('L')
                #     aaa.save('/media/NAS/nas_187/siwoo/2023/result/sam_offset_normoffset/x_{}_{}_mask.png')
                #
                    # plt.clf()
                    # plt.subplot(1, 3, 1)
                    # plt.imshow(mask[b].detach().cpu())
                    #
                    # plt.subplot(1, 3, 2)
                    # # plt.imshow(nuclei.detach().cpu())
                    # plt.imshow(ignored_map[b][0].detach().cpu())
                    #
                    # plt.subplot(1, 3, 3)
                    # plt.title(torch.sum(nuclei).detach().cpu().numpy())
                    # # plt.imshow(((self.x_map-x)*nuclei).detach().cpu().numpy())
                    # plt.imshow(gt[b][0].detach().cpu())
                    # plt.colorbar()
                    # plt.savefig('/media/NAS/nas_187/siwoo/2023/result/sam_ssl_ord/img/0/x_{}_{}_.png'.format(str(b), str(index)))
                    # print(aaa)
        if pred.shape[1] == 1:
            fg_loss = ((self.l1(pred, gt.float())*(mask != 0)/(mask != 0).sum())*ignored_map)
            bg_loss = ((self.l1(pred, gt.float()) * (mask == 0) / (mask == 0).sum())*ignored_map)
            offset_loss = torch.sum((fg_loss+bg_loss)/2+1e-7)*2
            # gt[ignored_map==True] = -1
            return offset_loss, gt

        elif pred.shape[1] == 2:
            # print(torch.unique(gt))
            # offset_loss = self.ce(pred, gt.long()).mean()
            offset_loss = self.l1(pred, gt.long()).mean()

            # fg_loss = ((self.l1(pred, gt.float()) * (mask != 0) / (mask != 0).sum()) * ignored_map)
            # bg_loss = ((self.l1(pred, gt.float()) * (mask == 0) / (mask == 0).sum()) * ignored_map)
            # offset_loss = torch.sum((fg_loss + bg_loss) / 2) * 2
            # gt[ignored_map==True] = -1
            return offset_loss, gt


        else:
            # print(self.ce(pred, gt_sonnet.long()).shape, pred.shape, gt_sonnet.shape, ignored_map.shape)
            offset_loss = torch.mean(torch.mean(self.l1(pred, gt_sonnet), dim=1)*ignored_map)
            return offset_loss, gt_sonnet

        # fg_loss = self.l1(pred, gt_sonnet.float()).sum(dim=1) * (mask != 0) / (mask != 0).sum()
        # bg_loss = self.l1(pred, gt_sonnet.float()).sum(dim=1) * (mask == 0) / (mask == 0).sum()
        # offset_loss = torch.sum((fg_loss+bg_loss)/2)*2

        #
        #
        # # offset_loss = cross_entropy(F.softmax(pred, dim=1), gt_sonnet)
        # # offset_loss = cross_entropy(F.softmax(pred, dim=1), gt_sonnet2)
        #
        # return offset_loss, gt_sonnet

def cross_entropy(softmax, y_target):
    loss = 0
    for j in range(len(softmax[0])):
        if j ==0:
            weight = 3
        elif j == 1:
            weight = 2
        else:
            weight = 1
        # loss += F.binary_cross_entropy(softmax[:,j], (y_target>j).float()) * weight / 8
        loss += -torch.mean(((softmax[:, j]>0.5)==(y_target[:,j]))*torch.log(softmax[:, j])) * weight
        # print(((softmax[:, j]>0.5)==y_target[:, j]), torch.sum(((softmax[:, j]>0.5)==y_target[:, j])))


    return loss
    # return - torch.sum(torch.log(softmax) * (y_target), dim=1)