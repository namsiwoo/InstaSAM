import argparse, os, random, sys
import skimage.io
import matplotlib.pyplot as plt

import numpy as np

from urllib.parse import urlparse
from cellpose import models, core
from cellpose import utils

import torch
from torch.utils.data import DataLoader
from datasets.MoNuSeg_dataset import Crop_dataset, DeepCell_dataset, MoNuSeg_weak_dataset, Galaxy_dataset, gt_with_weak_dataset
from utils.utils import accuracy_object_level, AJI_fast, average_precision, save_checkpoint, load_checkpoint, mk_colored, get_fast_pq

def split_forward(model, input, sam_input_size, device, num_hq_token, size=224):
    # size = 224
    overlap = 80

    b, c, h0, w0 = input.size()

    # zero pad for border patches
    pad_h = 0
    # here the padding is to make the the image size could be divided exactly by the size - overlap (the length of step)
    if h0 - size > 0 and (h0 - size) % (size - overlap) > 0:
        pad_h = (size - overlap) - (h0 - size) % (size - overlap)  # size is the the input size of model
        tmp = torch.zeros((b, c, pad_h, w0))
        input = torch.cat((input, tmp), dim=2)

    if w0 - size > 0 and (w0 - size) % (size - overlap) > 0:  # same as the above
        pad_w = (size - overlap) - (w0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        input = torch.cat((input, tmp), dim=3)

    _, c, h, w = input.size()

    output = torch.zeros((input.size(0), 1, h, w))
    offset_output = torch.zeros((input.size(0), num_hq_token, h, w))

    for i in range(0, h - overlap, size - overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h

        for j in range(0, w - overlap, size - overlap):
            c_end = j + size if j + size < w else w

            ind2_s = j + overlap // 2 if j > 0 else 0
            ind2_e = j + size - overlap // 2 if j + size < w else w

            input_patch = input[:, :, i:r_end, j:c_end]
            # input_patch = F.interpolate(input_patch, (sam_input_size, sam_input_size), mode='bilinear', align_corners=True)

            with torch.no_grad():
                # channels = [[2, 3], [0, 0], [0, 0]]
                channels = [1, 1]
                masks, flows, styles, diams = model.eval(input_patch, diameter=None, flow_threshold=None, channels=channels)
                masks = np.array(masks)
                flows = np.array(flows)
                styles = np.array(styles)
                diams = np.array(diams)
                print(masks.shape, flows.shape, styles.shape, diams.shape)

            output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = masks[:, :, ind1_s - i:ind1_e - i,
                                                     ind2_s - j:ind2_e - j]
            offset_output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = flows[:, :, ind1_s - i:ind1_e - i,
                                                     ind2_s - j:ind2_e - j]

    output = output[:, :, :h0, :w0].to(device)
    offset_output = offset_output[:, :, :h0, :w0].to(device)
    return output, offset_output

def test(args, device):
    if args.data_type == 'crop':
        test_dataseet = Crop_dataset(args, 'test', use_mask=args.sup, data=args.data)
    elif args.data_type == 'npy_c':
        test_dataseet = DeepCell_dataset(args, 'test', use_mask=args.sup, data='cell')
    elif args.data_type == 'npy_n':
        test_dataseet = DeepCell_dataset(args, 'test', use_mask=args.sup, data='nuclei')
    elif args.data_type == 'gal':
        test_dataseet = Galaxy_dataset(args, 'test_real', use_mask=args.sup, data='nuclei')
    else:
        test_dataseet = MoNuSeg_weak_dataset(args, 'test', sup=args.sup)

    test_dataloader = DataLoader(test_dataseet)

    from cellpose import models

    # DEFINE CELLPOSE MODEL
    # model_type='cyto3' or model_type='nuclei'
    model = models.Cellpose(gpu=device, model_type='cyto3')

    os.makedirs(os.path.join(args.result, 'img','test'), exist_ok=True)
    mean_dice, mean_iou, mean_aji = 0, 0, 0
    mean_dq, mean_sq, mean_pq = 0, 0, 0
    mean_ap1, mean_ap2, mean_ap3 = 0, 0, 0
    # if torch.distributed.get_rank() == 0:

    with torch.no_grad():
        for iter, pack in enumerate(test_dataloader):
            input = pack[0][0]
            mask = pack[0][1]
            if args.data == 'segpc':
                size = 1024
            else:
                size = 224

            img_name = pack[1][0]
            print(img_name, 'is processing....')



            output, output_offset = split_forward(model, input, size, device, 2, size)
            # binary_mask = torch.sigmoid(output).detach().cpu().numpy()
            print(output.shape, output_offset.shape)
            print(torch.unique(output))


            if len(np.unique(binary_map)) == 1:
                dice, iou, aji = 0, 0, 0
            else:
                dice, iou = accuracy_object_level(instance_map, mask[0][0].detach().cpu().numpy())
                aji = AJI_fast(mask[0][0].detach().cpu().numpy(), instance_map, img_name)
                pq_list, _ = get_fast_pq(mask[0][0].detach().cpu().numpy(), instance_map) #[dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]
                ap, _, _, _ = average_precision(mask[0][0].detach().cpu().numpy(), instance_map)

            mean_dice += dice / (len(test_dataloader))  # *len(local_rank))
            mean_iou += iou / (len(test_dataloader))  # len(local_rank))
            mean_aji += aji / (len(test_dataloader))

            mean_dq += pq_list[0] / (len(test_dataloader))  # *len(local_rank))
            mean_sq += pq_list[1] / (len(test_dataloader))  # len(local_rank))
            mean_pq += pq_list[2] / (len(test_dataloader))

            mean_ap1 += ap[0] / (len(test_dataloader))
            mean_ap2 += ap[1] / (len(test_dataloader))
            mean_ap3 += ap[2] / (len(test_dataloader))

            instance_map = mk_colored(instance_map) * 255
            instance_map = Image.fromarray((instance_map).astype(np.uint8))
            instance_map.save(os.path.join(args.result, 'img', 'test', str(img_name) + '_pred_inst.png'))

            marker = mk_colored(marker) * 255
            marker = Image.fromarray((marker).astype(np.uint8))
            marker.save(os.path.join(args.result, 'img', 'test', str(img_name) + '_marker.png'))

            pred = mk_colored(binary_map) * 255
            pred = Image.fromarray((pred).astype(np.uint8))
            pred.save(os.path.join(args.result, 'img', 'test', str(img_name) + '_pred.png'))

            pred_flow_vis = Image.fromarray(pred_flow_vis.astype(np.uint8))
            pred_flow_vis.save(os.path.join(args.result, 'img', 'test', str(img_name) + '_flow_vis.png'))

            mask = mk_colored(mask[0][0].detach().cpu().numpy()) * 255
            mask = Image.fromarray((mask).astype(np.uint8))
            mask.save(os.path.join(args.result, 'img', 'test', str(img_name) + '_mask.png'))



    print('test result: Average- Dice\tIOU\tAJI: '
                 '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_dice, mean_iou, mean_aji))
    print('test result: Average- DQ\tSQ\tPQ: '
                 '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_dq, mean_sq, mean_pq))
    print('test result: Average- AP1\tAP2\tAP3: '
                 '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_ap1, mean_ap2, mean_ap3))


    f = open(os.path.join(args.result,'img', 'test', "result.txt"), 'w')
    f.write('***test result_mask*** Average- Dice\tIOU\tAJI: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_dice, mean_iou, mean_aji))
    f.write('***test result_mask*** Average- DQ\tSQ\tPQ: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_dq, mean_sq, mean_pq))
    f.write('***test result_mask*** Average- AP1\tAP2\tAP3: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_ap1, mean_ap2, mean_ap3))
    f.close()

    f = open(os.path.join(args.result, "result.txt"), 'w')
    f.write('***test result_mask*** Average- Dice\tIOU\tAJI: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_dice, mean_iou, mean_aji))
    f.write('***test result_mask*** Average- DQ\tSQ\tPQ: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_dq, mean_sq, mean_pq))
    f.write('***test result_mask*** Average- AP1\tAP2\tAP3: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_ap1, mean_ap2, mean_ap3))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data_type', default='crop', type=str, help='crop, patch')
    parser.add_argument('--data', default='MoNuSeg', help='MoNuSeg, CPM 17, CoNSeP, TNBC')
    parser.add_argument('--shift', default='0', type=int, help='0, 2, 4, 6, 8')
    parser.add_argument('--fs', action='store_true', help='few-shot setting')
    parser.add_argument('--data_path', default='', help='few-shot setting')
    parser.add_argument('--ck_point', default=None, type=str, help='MoNuSeg, CPM 17, CoNSeP, TNBC')
    parser.add_argument('--sup', action='store_true')
    parser.add_argument('--result', default='/media/NAS/nas_187/siwoo/2024/revision/cellpose_model/cellpose', help='')

    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    if args.data == "MoNuSeg":
        if args.shift == 0:
            args.data_path = '/media/NAS/nas_70/open_dataset/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg'
        else:
            args.data_path = '/media/NAS/nas_70/open_dataset/MoNuSeg/MoNuSeg_shift{:s}/via instance learning data_for_train/MoNuSeg_shift{:s}'.format(args.shift)
    elif args.data == "CPM":
        if args.shift == 0:
            args.data_path = '/media/NAS/nas_70/open_dataset/CPM/CPM 17/via instance learning data_for_train/CPM 17'
        else:
            args.data_path = '/media/NAS/nas_70/open_dataset/CPM/CPM 17_shift{:s}/via instance learning data_for_train/CPM 17_shift{:s}'.format(args.shift)
    elif args.data == "CoNSeP":
        if args.shift == 0:
            args.data_path = '/media/NAS/nas_70/open_dataset/CoNSeP/CoNSeP/via instance learning data_for_train/CoNSeP'
    elif args.data == "TNBC":
        if args.shift == 0:
            args.data_path = '/media/NAS/nas_70/open_dataset/TNBC/TNBC_new/via instance learning data_for_train/TNBC_new'
        else:
            args.data_path = 'not yet'
    elif args.data == "segpc":
        if args.shift == 0:
            args.data_path = '/media/NAS/nas_70/open_dataset/segpc/segpc'
        else:
            args.data_path = 'not yet'
    elif args.data == 'pannuke':
        if args.shift == 0:
            args.data_path = '/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch'
    elif args.data == 'cellpose':
        args.data_path = '/media/NAS/nas_70/open_dataset/Cellpose'
    elif args.data == 'DeepCell':
        args.data_path = '/media/NAS/nas_70/open_dataset/DeepCell'
    elif args.data == 'galaxy':
        args.data_path = '/media/NAS/nas_187/datasets/galaxy_dataset_UNIST'
    else:
        print('wrong data name was entered')

    print('=' * 40)
    print(' ' * 14 + 'Arguments')
    for arg in sorted(vars(args)):
        print(arg + ':', getattr(args, arg))
    print('=' * 40)


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    test(args, device)