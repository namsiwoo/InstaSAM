import torch
import os, random, gc
import argparse
from PIL import Image
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets.MoNuSeg_dataset import Crop_dataset, DeepCell_dataset, MoNuSeg_weak_dataset, Galaxy_dataset, gt_with_weak_dataset, IHC_dataset

import models

from utils.utils import accuracy_object_level, AJI_fast, average_precision, save_checkpoint, load_checkpoint, mk_colored, get_fast_pq
from utils.vis_flow import flow_to_color
from utils.hv_process import make_instance_hv, make_instance_sonnet, make_instance_marker

def split_forward(sam_model, input, sam_input_size, device, num_hq_token, size=224):
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
            input_patch = F.interpolate(input_patch, (sam_input_size, sam_input_size), mode='bilinear', align_corners=True)

            with torch.no_grad():
                pred, offset_pred = sam_model.infer(input_patch.to(device))
                pred = F.interpolate(pred, (size, size), mode='bilinear', align_corners=True)
                offset_pred = F.interpolate(offset_pred, (size, size), mode='bilinear', align_corners=True)


            output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = pred[:, :, ind1_s - i:ind1_e - i,
                                                     ind2_s - j:ind2_e - j]
            offset_output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = offset_pred[:, :, ind1_s - i:ind1_e - i,
                                                     ind2_s - j:ind2_e - j]

    output = output[:, :, :h0, :w0].to(device)
    offset_output = offset_output[:, :, :h0, :w0].to(device)
    return output, offset_output


def main(args):
    f = open(os.path.join(args.result, 'log.txt'), 'w')
    f.write('=' * 40)
    f.write('Arguments')
    f.write(str(args))
    f.write('=' * 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    #adapter
    if args.model_type == 'vit_h':
        encoder_mode = {'name': 'sam', 'img_size': args.img_size, 'mlp_ratio': 4, 'patch_size': 16, 'qkv_bias': True, 'use_rel_pos': True, 'window_size': 14, 'out_chans': 256, 'scale_factor': 32, 'input_type': 'fft',
                        'freq_nums': 0.25, 'prompt_type': 'highpass', 'prompt_embed_dim': 256, 'tuning_stage': 1234, 'handcrafted_tune': True, 'embedding_tune': True, 'adaptor': 'adaptor', 'embed_dim': 1280,
                        'depth': 32, 'num_heads': 16, 'global_attn_indexes': [7, 15, 23, 31]}
        sam_checkpoint = '/media/NAS/nas_187/siwoo/2023/SAM model/SAM-Adapter-PyTorch-main/sam_vit_h_4b8939.pth'
        # sam_checkpoint = '/media/NAS/nas_187/siwoo/2023/result/transformer_freeze_new_h2_pseudo_MO_2/model/Aji_best_model.pth'
    elif args.model_type == 'vit_l':
        encoder_mode = {'name': 'sam', 'img_size': args.img_size, 'mlp_ratio': 4, 'patch_size': 16, 'qkv_bias': True, 'use_rel_pos': True, 'window_size': 14, 'out_chans': 256, 'scale_factor': 32, 'input_type': 'fft',
                        'freq_nums': 0.25, 'prompt_type': 'highpass', 'prompt_embed_dim': 256, 'tuning_stage': 1234, 'handcrafted_tune': True, 'embedding_tune': True, 'adaptor': 'adaptor', 'embed_dim': 1024,
                        'depth': 24, 'num_heads': 16, 'global_attn_indexes': [5, 11, 17, 23]}
        sam_checkpoint = '/media/NAS/nas_187/siwoo/2023/SAM model/SAM-Adapter-PyTorch-main/sam_vit_l_0b3195.pth'
    elif args.model_type == 'vit_b' or args.model_type == 'medsam':
        encoder_mode = {'name': 'sam', 'img_size': args.img_size, 'mlp_ratio': 4, 'patch_size': 16, 'qkv_bias': True, 'use_rel_pos': True, 'window_size': 14, 'out_chans': 256, 'scale_factor': 32, 'input_type': 'fft',
                        'freq_nums': 0.25, 'prompt_type': 'highpass', 'prompt_embed_dim': 256, 'tuning_stage': 1234, 'handcrafted_tune': True, 'embedding_tune': True, 'adaptor': 'adaptor', 'embed_dim': 768,
                        'depth': 12, 'num_heads': 12, 'global_attn_indexes': [2, 5, 8, 11]}
        if args.model_type == 'vit_b':
            sam_checkpoint = '/media/NAS/nas_187/siwoo/2023/SAM model/SAM-Adapter-PyTorch-main/sam_vit_b_01ec64.pth'
        else:
            sam_checkpoint = '/media/NAS/nas_187/siwoo/2023/SAM model/SAM-Adapter-PyTorch-main/medsam_vit_b.pth'

    sam_model = models.sam.SAM(inp_size=1024, encoder_mode=encoder_mode, loss='iou', device=device)
    sam_model.optimizer = torch.optim.AdamW(sam_model.parameters(), lr=args.lr)
    if args.auto_cast:
        sam_model.scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sam_model.optimizer, 20, eta_min=1.0e-7)
    sam_model = load_checkpoint(sam_model, sam_checkpoint)

    for name, para in sam_model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
        if "prompt_encoder" in name:
            para.requires_grad_(False)
        if "mask_decoder" in name:
            para.requires_grad_(False)

    sam_model.make_HQ_model(model_type=args.model_type, num_token=args.num_hq_token)
    if args.adapter2:
        sam_model.make_adapter2()
    if args.resume != 0:
        sam_model = load_checkpoint(sam_model, os.path.join(args.result, 'model', str(args.resume)+'_model.pth'))
    if args.ck_point is not None:
        sam_model = load_checkpoint(sam_model, args.ck_point)

    sam_model = sam_model.cuda()



    print(torch.cuda.memory_allocated() / 1024 / 1024, '******')
    model_total_params = sum(p.numel() for p in sam_model.parameters())
    model_grad_params = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))
    # for name, p in sam_model.named_parameters():
    #     if p.requires_grad:
    #         print('========', name)
    #     else:
    #         print('********', name)


    train_dataset = gt_with_weak_dataset(args, 'train', semi=args.semi)
    val_dataset = gt_with_weak_dataset(args, 'val', semi=args.semi)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset)

    max_Dice, max_Aji = 0, 0
    total_train_loss = []
    for epoch in range(args.epochs):
        if args.resume != 0:
            epoch += args.resume

    # if (epoch + 1) % 20 == 0:
    #     sam_model = load_checkpoint(sam_model, os.path.join(args.model, str(epoch)+ '_model.pth'))

        os.makedirs(os.path.join(args.result, 'img', str(epoch)), exist_ok=True)
        sam_model.train()
        train_loss = 0

        for iter, batch in enumerate(train_dataloader): # batch[0]
            img = batch[0][0]
            img_name = batch[1][0]
            if args.semi == True:
                label = batch[0][1].squeeze(1)
                point = batch[0][2]
                sam_model.set_input(img, label)
                low_res_masks, hq_mask, bce_loss, offset_loss, iou_loss, offset_gt, bce_local_loss, iou_local_loss = sam_model.optimize_parameters_semi(
                    point, os.path.join(args.result, 'img', str(epoch), img_name + '.png'), args.semi, epoch, args.auto_cast)
            else:
                label = batch[0][1].squeeze(1)
                point = batch[0][2]

                sam_model.set_input(img)
                low_res_masks, hq_mask, bce_loss, offset_loss, iou_loss, offset_gt, bce_local_loss, iou_local_loss, f_loss = sam_model.optimize_parameters_semi(
                    point, os.path.join(args.result, 'img', str(epoch), img_name + '.png'), args.semi, epoch, label, args.auto_cast)  # point, epoch, batch[1][0]
            # clu_label = batch[0][3].squeeze(1)
            # vor_label = batch[0][4].squeeze(1)


            # bce_loss, offset_gt, offset_loss, iou_loss = backwards
            # bce_local_loss, iou_local_loss = backwards_local
            lr_scheduler.step()

            loss = bce_loss + iou_loss + 5*offset_loss + bce_local_loss + iou_local_loss + f_loss #sam_model.loss_G.item()

            train_loss += loss / len(train_dataloader)

            if (iter + 1) % args.print_fq == 0:
                print('{}/{} epoch, {}/{} batch, train loss: {}, bce: {}, iou: {}, offset: {} // local bce: {} iou: {} // f_loss: {}'.format(epoch,
                                                                                                        args.epochs,
                                                                                                        iter + 1,
                                                                                                        len(train_dataloader),
                                                                                                        loss, bce_loss,
                                                                                                        iou_loss,
                                                                                                        offset_loss, bce_local_loss, iou_local_loss, f_loss))

                if args.plt == True:
                    import matplotlib.pyplot as plt

                    def norm(img):
                        return (img - np.min(img)) / (np.max(img) - np.min(img))

                    plt.clf()
                    img1 = norm(batch[0][0].detach().cpu().numpy()[0].transpose(1, 2, 0))
                    if args.num_hq_token == 1:
                        plt.subplot(1, 5, 1)
                        plt.imshow(img1)
                        plt.subplot(1, 5, 2)
                        plt.imshow(low_res_masks.detach().cpu().numpy()[0, 0])

                        from scipy.ndimage.morphology import binary_dilation
                        point = binary_dilation(point.detach().cpu().numpy()[0, 0], iterations=2)
                        plt.subplot(1, 5, 3)
                        plt.imshow(point)

                        plt.subplot(1, 5, 4)
                        plt.imshow(offset_gt.detach().cpu().numpy()[0])
                        plt.colorbar()
                        plt.subplot(1, 5, 5)
                        plt.imshow(hq_mask.detach().cpu().numpy()[0, 1])
                    else:
                        plt.subplot(2, 4, 1)
                        plt.imshow(img1)
                        plt.subplot(2, 4, 2)
                        # plt.imshow(label.detach().cpu().numpy()[0])
                        plt.imshow(low_res_masks.detach().cpu().numpy()[0, 0])

                        if args.num_hq_token >= 3:
                            plt.subplot(2, 4, 3)
                            # plt.imshow(low_res_masks.detach().cpu().numpy()[0,0])
                            plt.imshow(hq_mask.detach().cpu().numpy()[0, -1])
                            plt.colorbar()

                            plt.subplot(2, 4, 4)
                            plt.imshow(offset_gt.detach().cpu().numpy()[0][-1])
                            plt.colorbar()

                        # plt.subplot(2, 4, 3)
                        # plt.imshow(offset_gt.detach().cpu().numpy()[0][0]>1)
                        # plt.colorbar()
                        #
                        # plt.subplot(2, 4, 4)
                        # plt.imshow(offset_gt.detach().cpu().numpy()[0][0]<-1)
                        # plt.colorbar()

                        plt.subplot(2, 4, 5)
                        plt.imshow(hq_mask.detach().cpu().numpy()[0, 0])
                        plt.colorbar()

                        plt.subplot(2, 4, 6)
                        plt.imshow(hq_mask.detach().cpu().numpy()[0, 1])
                        plt.colorbar()

                        plt.subplot(1, 4, 3)
                        # plt.imshow(point.numpy()[0][0])
                        plt.imshow(offset_gt.detach().cpu().numpy()[0][0])
                        plt.colorbar()

                        plt.subplot(1, 4, 4)
                        plt.imshow(offset_gt.detach().cpu().numpy()[0][1])
                        plt.colorbar()



                        # def colorize(ch, vmin=None, vmax=None):
                        #     """Will clamp value value outside the provided range to vmax and vmin."""
                        #     cmap = plt.get_cmap("jet")
                        #     ch = np.squeeze(ch.astype("float32"))
                        #     vmin = vmin if vmin is not None else ch.min()
                        #     vmax = vmax if vmax is not None else ch.max()
                        #     ch[ch > vmax] = vmax  # clamp value
                        #     ch[ch < vmin] = vmin
                        #     ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
                        #     # take RGB from RGBA heat map
                        #     ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
                        #     return ch_cmap
                        #
                        # aaa = torch.sigmoid(low_res_masks).detach().cpu().numpy()[0, 0]
                        # aaa = (aaa-np.min(aaa))/(np.max(aaa)-np.min(aaa))
                        # aaa = Image.fromarray((aaa*255).astype(np.uint8))
                        # aaa.save(os.path.join(args.result, str(epoch), img_name+ '_binary.png'))
                        #
                        # aaa = hq_mask.detach().cpu().numpy()[0, 1]
                        # aaa = colorize(aaa)
                        # aaa = Image.fromarray(aaa)
                        # aaa.save(os.path.join(args.result, str(epoch), img_name+ '_h.png'))
                        #
                        # aaa = hq_mask.detach().cpu().numpy()[0, 0]
                        # aaa = colorize(aaa)
                        # aaa = Image.fromarray(aaa)
                        # aaa.save(os.path.join(args.result, str(epoch), img_name+ '_v.png'))
                        #
                        # binary_map, instance_map, marker = make_instance_hv(torch.sigmoid(low_res_masks)[0][0].detach().cpu().numpy(),
                        #                                                     hq_mask[0].detach().cpu().numpy())
                        # instance_map = mk_colored(instance_map) * 255
                        # instance_map = Image.fromarray((instance_map).astype(np.uint8))
                        # instance_map.save(os.path.join(args.result, str(epoch), img_name+ '_inst.png'))

                    plt.savefig(os.path.join(args.result, 'img', str(epoch), str(iter) + 'ex.png'))

            gc.collect()
            torch.cuda.empty_cache()
        total_train_loss.append(train_loss)
        print('{} epoch, mean train loss: {}'.format(epoch, total_train_loss[-1]))

        # if epoch % 5 == 0:
        #     save_checkpoint(os.path.join(args.result, 'model', str(epoch) + '_model.pth'), sam_model, epoch)

        if epoch >= args.start_val:
            sam_model.eval()
            mean_dice, mean_iou, mean_aji = 0, 0, 0

            with torch.no_grad():
                for iter, pack in enumerate(val_dataloader):
                    input = pack[0][0]
                    mask = pack[0][1]
                    img_name = pack[1][0]
                    if args.data == 'segpc':
                        size = 1024
                    else:
                        size = 224

                    output, output_offset = split_forward(sam_model, input, args.img_size, device, args.num_hq_token, size)
                    binary_mask = torch.sigmoid(output).detach().cpu().numpy()

                    if args.num_hq_token == 2:
                        pred_flow_vis = flow_to_color(output_offset[0].detach().cpu().numpy().transpose(1, 2, 0))
                        binary_map, instance_map, marker = make_instance_hv(binary_mask[0][0],
                                                                            output_offset[0].detach().cpu().numpy())
                    elif args.num_hq_token == 1:
                        pred_flow_vis = output_offset[0][1].detach().cpu().numpy()
                        pred_flow_vis = (pred_flow_vis-np.min(pred_flow_vis))/(np.max(pred_flow_vis)-np.min(pred_flow_vis))*255
                        binary_map, instance_map, marker = make_instance_marker(binary_mask[0][0], output_offset[0][
                            1].detach().cpu().numpy(), args.ord_th)
                    else:
                        bg = torch.zeros(1, 1, 1000, 1000) + args.ord_th  # K 0.15
                        bg = bg.to(device)
                        output_offset = torch.argmax(torch.cat([bg, output_offset], dim=1), dim=1)
                        pred_flow_vis = ((output_offset[0].detach().cpu().numpy() * 255) / 9).astype(np.uint8)
                        binary_map, instance_map, marker = make_instance_sonnet(binary_mask[0][0],
                                                                                output_offset[0].detach().cpu().numpy())

                    if len(np.unique(binary_map)) == 1:
                        dice, iou, aji = 0, 0, 0
                    else:
                        # dice, iou = accuracy_object_level(instance_map, mask[0][0].detach().cpu().numpy())
                        # aji = AJI_fast(mask[0][0].detach().cpu().numpy(), instance_map)
                        dice, iou = accuracy_object_level(instance_map, mask[0][0].detach().cpu().numpy())
                        aji = AJI_fast(mask[0][0].detach().cpu().numpy(), instance_map, img_name)

                    mean_dice += dice / len(val_dataloader)  # *len(local_rank))
                    mean_iou += iou / len(val_dataloader)  # len(local_rank))
                    mean_aji += aji / len(val_dataloader)  # *len(local_rank))
                    # print(len(val_dataloader), mean_dice, mean_aji)

                    instance_map = mk_colored(instance_map) * 255
                    instance_map = Image.fromarray((instance_map).astype(np.uint8))
                    instance_map.save(os.path.join(args.result, 'img', str(epoch), str(img_name) + '_pred_inst.png'))

                    marker = mk_colored(marker) * 255
                    marker = Image.fromarray((marker).astype(np.uint8))
                    marker.save(os.path.join(args.result, 'img', str(epoch), str(img_name) + '_marker.png'))

                    binary_map = mk_colored(binary_map) * 255
                    binary_map = Image.fromarray((binary_map).astype(np.uint8))
                    binary_map.save(os.path.join(args.result, 'img', str(epoch), str(img_name) + '_pred.png'))

                    pred_flow_vis = Image.fromarray(pred_flow_vis.astype(np.uint8))
                    pred_flow_vis.save(os.path.join(args.result, 'img', str(epoch), str(img_name) + '_flow_vis.png'))

                    mask = mk_colored(mask[0][0].detach().cpu().numpy()) * 255
                    mask = Image.fromarray((mask).astype(np.uint8))
                    mask.save(os.path.join(args.result, 'img', str(epoch), str(img_name) + '_mask.png'))

                    del binary_map, instance_map, marker, pred_flow_vis, mask, input

                f = open(os.path.join(args.result, 'img', str(epoch), "result.txt"), 'w')
                f.write('***test result_mask*** Average- Dice\tIOU\tAJI: '
                        '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_dice, mean_iou, mean_aji))
                f.close()

                if max_Dice < mean_dice:
                    print('save {} epoch!!--Dice: {}'.format(str(epoch), mean_dice))
                    save_checkpoint(os.path.join(args.result, 'model', 'Dice_best_model.pth'), sam_model, epoch)
                    max_Dice = mean_dice

                if max_Aji < mean_aji:
                    print('save {} epoch!!--Aji: {}'.format(str(epoch), mean_aji))
                    save_checkpoint(os.path.join(args.result, 'model', 'Aji_best_model.pth'), sam_model, epoch)
                    max_Aji = mean_aji

                print(epoch, ': Average- Dice\tIOU\tAJI: '
                             '\t\t{:.4f}\t{:.4f}\t{:.4f} (b Dice: {}, b Aji: {})'.format(mean_dice, mean_iou,
                                                                                         mean_aji, max_Dice, max_Aji))


def test(args, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # sam_model = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    # sam_model = sam_model.to(device)

    # adapter
    if args.model_type == 'vit_h':
        encoder_mode = {'name': 'sam', 'img_size': args.img_size, 'mlp_ratio': 4, 'patch_size': 16, 'qkv_bias': True, 'use_rel_pos': True, 'window_size': 14, 'out_chans': 256, 'scale_factor': 32, 'input_type': 'fft',
                        'freq_nums': 0.25, 'prompt_type': 'highpass', 'prompt_embed_dim': 256, 'tuning_stage': 1234, 'handcrafted_tune': True, 'embedding_tune': True, 'adaptor': 'adaptor', 'embed_dim': 1280,
                        'depth': 32, 'num_heads': 16, 'global_attn_indexes': [7, 15, 23, 31]}
    elif args.model_type == 'vit_l':
        encoder_mode = {'name': 'sam', 'img_size': args.img_size, 'mlp_ratio': 4, 'patch_size': 16, 'qkv_bias': True, 'use_rel_pos': True, 'window_size': 14, 'out_chans': 256, 'scale_factor': 32, 'input_type': 'fft',
                        'freq_nums': 0.25, 'prompt_type': 'highpass', 'prompt_embed_dim': 256, 'tuning_stage': 1234, 'handcrafted_tune': True, 'embedding_tune': True, 'adaptor': 'adaptor', 'embed_dim': 1024,
                        'depth': 24, 'num_heads': 16, 'global_attn_indexes': [5, 11, 17, 23]}
    elif args.model_type == 'vit_b':
        encoder_mode = {'name': 'sam', 'img_size': args.img_size, 'mlp_ratio': 4, 'patch_size': 16, 'qkv_bias': True,
                        'use_rel_pos': True, 'window_size': 14, 'out_chans': 256, 'scale_factor': 32,
                        'input_type': 'fft',
                        'freq_nums': 0.25, 'prompt_type': 'highpass', 'prompt_embed_dim': 256, 'tuning_stage': 1234,
                        'handcrafted_tune': True, 'embedding_tune': True, 'adaptor': 'adaptor', 'embed_dim': 768,
                        'depth': 12, 'num_heads': 12, 'global_attn_indexes': [2, 5, 8, 11]}
        sam_checkpiont = 'sam_vit_b_01ec64.pth'

    sam_model = models.sam.SAM(inp_size=1024, encoder_mode=encoder_mode, loss='iou', device=device)

    sam_model.make_HQ_model(model_type=args.model_type, num_token=args.num_hq_token)
    if args.adapter2:
        sam_model.make_adapter2()
    sam_model = sam_model.cuda()

    # sam_checkpoint = torch.load(os.path.join(args.model, 'Aji_best_model.pth'))
    # sam_model.load_state_dict(sam_checkpoint, strict=False)
    sam_model = load_checkpoint(sam_model, os.path.join(args.result, 'model', 'Aji_best_model.pth'))
    # sam_model = load_checkpoint(sam_model, os.path.join(args.model, 'Dice_best_model.pth'))

    test_dataset = gt_with_weak_dataset(args, 'test', semi=args.semi)

    test_dataloader = DataLoader(test_dataset)
    if args.test_name == "None":
        args.test_name == 'test'
    else:
        args.test_name = 'test_'+args.test_name

    os.makedirs(os.path.join(args.result, 'img',args.test_name), exist_ok=True)
    sam_model.eval()
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

            output, output_offset = split_forward(sam_model, input, args.img_size, device, args.num_hq_token, size)
            binary_mask = torch.sigmoid(output).detach().cpu().numpy()

            if args.num_hq_token == 2:
                pred_flow_vis = flow_to_color(output_offset[0].detach().cpu().numpy().transpose(1, 2, 0))
                binary_map, instance_map, marker = make_instance_hv(binary_mask[0][0],
                                                                    output_offset[0].detach().cpu().numpy())
            elif args.num_hq_token == 1:
                pred_flow_vis = output_offset[0][1].detach().cpu().numpy() * 255
                binary_map, instance_map, marker = make_instance_marker(binary_mask[0][0], output_offset[0][
                    1].detach().cpu().numpy(), args.ord_th)
            else:
                bg = torch.zeros(1, 1, 1000, 1000) + args.ord_th  # K 0.15
                bg = bg.to(device)
                output_offset = torch.argmax(torch.cat([bg, output_offset], dim=1), dim=1)
                pred_flow_vis = ((output_offset[0].detach().cpu().numpy() * 255) / 9).astype(np.uint8)
                binary_map, instance_map, marker = make_instance_sonnet(binary_mask[0][0],
                                                                        output_offset[0].detach().cpu().numpy())

            if len(np.unique(binary_map)) == 1:
                dice, iou, aji = 0, 0, 0
                pq_list = [0, 0, 0]
                ap = [0, 0, 0]
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
            instance_map.save(os.path.join(args.result, 'img', args.test_name, str(img_name) + '_pred_inst.png'))

            marker = mk_colored(marker) * 255
            marker = Image.fromarray((marker).astype(np.uint8))
            marker.save(os.path.join(args.result, 'img', args.test_name, str(img_name) + '_marker.png'))

            pred = mk_colored(binary_map) * 255
            pred = Image.fromarray((pred).astype(np.uint8))
            pred.save(os.path.join(args.result, 'img', args.test_name, str(img_name) + '_pred.png'))

            pred_flow_vis = Image.fromarray(pred_flow_vis.astype(np.uint8))
            pred_flow_vis.save(os.path.join(args.result, 'img', args.test_name, str(img_name) + '_flow_vis.png'))

            mask = mk_colored(mask[0][0].detach().cpu().numpy()) * 255
            mask = Image.fromarray((mask).astype(np.uint8))
            mask.save(os.path.join(args.result, 'img', args.test_name, str(img_name) + '_mask.png'))



    print('test result: Average- Dice\tIOU\tAJI: '
                 '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_dice, mean_iou, mean_aji))
    print('test result: Average- DQ\tSQ\tPQ: '
                 '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_dq, mean_sq, mean_pq))
    print('test result: Average- AP1\tAP2\tAP3: '
                 '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_ap1, mean_ap2, mean_ap3))


    f = open(os.path.join(args.result,'img', args.test_name, "result.txt"), 'w')
    f.write('***test result_mask*** Average- Dice\tIOU\tAJI: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_dice, mean_iou, mean_aji))
    f.write('***test result_mask*** Average- DQ\tSQ\tPQ: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_dq, mean_sq, mean_pq))
    f.write('***test result_mask*** Average- AP1\tAP2\tAP3: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_ap1, mean_ap2, mean_ap3))
    f.close()

    f = open(os.path.join(args.result, "result"+args.test_name[4:]+".txt"), 'w')
    f.write('***test result_mask*** Average- Dice\tIOU\tAJI: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_dice, mean_iou, mean_aji))
    f.write('***test result_mask*** Average- DQ\tSQ\tPQ: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_dq, mean_sq, mean_pq))
    f.write('***test result_mask*** Average- AP1\tAP2\tAP3: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_ap1, mean_ap2, mean_ap3))
    f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--resume', default=0, type=int, help='')
    parser.add_argument('--start_val', default=30, type=int)
    parser.add_argument('--plt', action='store_true')
    parser.add_argument('--semi', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--adapter2',action='store_true')
    parser.add_argument('--auto_cast',action='store_true')

    parser.add_argument('--data_type',default='crop', type=str ,help='crop, patch')
    parser.add_argument('--data',default='MoNuSeg',help='MoNuSeg, CPM 17, CoNSeP, TNBC')
    parser.add_argument('--shift',default='0', type=int, help='0, 2, 4, 6, 8')
    parser.add_argument('--fs', action='store_true', help='few-shot setting')
    parser.add_argument('--data_path', default='', help='few-shot setting')
    parser.add_argument('--ck_point',default=None, type=str, help='MoNuSeg, CPM 17, CoNSeP, TNBC')


    # parser.add_argument('--data', default='/media/NAS/nas_32/siwoo/TNBC/TNBC/via instance learning data_for_train/TNBC', help='path to dataset')
    # parser.add_argument('--data', default='/media/NAS/nas_32/siwoo/TNBC/TNBC_shift4/via instance learning data_for_train/TNBC_shift4', help='path to dataset')
    # parser.add_argument('--data', default='/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch')
    # parser.add_argument('--data', default='/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC')

    parser.add_argument('--img_size', default=1024, help='')
    parser.add_argument('--num_hq_token', default=2, type=int, help='')
    parser.add_argument('--epochs', default=100, type=int, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--model_type', default='vit_h', help='')
    # parser.add_argument('--sam_checkpoint', default='/media/NAS/nas_187/siwoo/2023/result/transformer_freeze_new_h2_MO/model/Aji_best_model.pth')
    parser.add_argument('--print_fq', default=15, type=int, help='')
    parser.add_argument('--ord_th', default=0.5, type=float)

    # parser.add_argument('--result', default='/media/NAS/nas_187/siwoo/2023/result/MO_shift4_fs2/img', help='')
    # parser.add_argument('--model', default='/media/NAS/nas_187/siwoo/2023/result/MO_shift_4_fs2/model', help='')
    parser.add_argument('--result', default='/media/NAS/nas_187/siwoo/2024/revision/cellpose', help='')
    parser.add_argument('--test_name', default='None', type=str, help='')

    # parser.add_argument('--result', default='/media/NAS/nas_187/siwoo/2024/revision/pannuke_sup/img', help='')
    # parser.add_argument('--model', default='/media/NAS/nas_187/siwoo/2024/revision/pannuke_sup/model', help='')
    # parser.add_argument('--result', default='/media/NAS/nas_187/siwoo/2023/best/CPM_samups_5offset_shift8_2/img', help='')
    # parser.add_argument('--model', default='/media/NAS/nas_187/siwoo/2023/best/CPM_samups_5offset_shift8_2/model', help='')

    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()


    if args.result != None:
        os.makedirs(os.path.join(args.result, 'img'), exist_ok=True)
        os.makedirs(os.path.join(args.result, 'model'), exist_ok=True)

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
            args.data_path = '/media/NAS/nas_70/open_dataset/TNBC/TNBC/via instance learning data_for_train/TNBC'
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
    elif args.data == 'DeepLIIF':
        args.data_path = '/media/NAS/nas_70/open_dataset/DeepLIIF/DeepLIIF'
    elif args.data == 'DeepLIIF_BC':
        args.data_path = '/media/NAS/nas_70/open_dataset/DeepLIIF/BC'
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

    if args.train==True:
        main(args)
    if args.test==True:
        test(args, device)

