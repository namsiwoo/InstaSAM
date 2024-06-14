import torch
import os, random, gc
import argparse
from PIL import Image
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets.MoNuSeg_dataset import MoNuSeg_weak_dataset

from models import SAM_allinsam

from utils.utils import accuracy_object_level, AJI_fast, save_checkpoint, load_checkpoint, mk_colored
from utils.vis_flow import flow_to_color
from utils.hv_process import make_instance_hv, make_instance_sonnet, make_instance_marker

def split_forward(sam_model, input, mask, sam_input_size, device, num_hq_token):
    size = 224
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
                pred = sam_model.infer(input_patch.to(device))


            output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = pred[:, :, ind1_s - i:ind1_e - i,
                                                     ind2_s - j:ind2_e - j]

    output = output[:, :, :h0, :w0].to(device)
    return output


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
        sam_checkpiont = 'sam_vit_h_4b8939.pth'
        # sam_checkpiont = '/media/NAS/nas_187/siwoo/2023/result/transformer_freeze_new_h2_pseudo_MO_2/model/Aji_best_model.pth'
    elif args.model_type == 'vit_l':
        encoder_mode = {'name': 'sam', 'img_size': args.img_size, 'mlp_ratio': 4, 'patch_size': 16, 'qkv_bias': True, 'use_rel_pos': True, 'window_size': 14, 'out_chans': 256, 'scale_factor': 32, 'input_type': 'fft',
                        'freq_nums': 0.25, 'prompt_type': 'highpass', 'prompt_embed_dim': 256, 'tuning_stage': 1234, 'handcrafted_tune': True, 'embedding_tune': True, 'adaptor': 'adaptor', 'embed_dim': 1024,
                        'depth': 24, 'num_heads': 16, 'global_attn_indexes': [5, 11, 17, 23]}
        sam_checkpiont = 'sam_vit_l_0b3195.pth'
    elif args.model_type == 'vit_b':
        encoder_mode = {'name': 'sam', 'img_size': args.img_size, 'mlp_ratio': 4, 'patch_size': 16, 'qkv_bias': True, 'use_rel_pos': True, 'window_size': 14, 'out_chans': 256, 'scale_factor': 32, 'input_type': 'fft',
                        'freq_nums': 0.25, 'prompt_type': 'highpass', 'prompt_embed_dim': 256, 'tuning_stage': 1234, 'handcrafted_tune': True, 'embedding_tune': True, 'adaptor': 'adaptor', 'embed_dim': 768,
                        'depth': 12, 'num_heads': 12, 'global_attn_indexes': [2, 5, 8, 11]}
        sam_checkpiont = 'sam_vit_b_01ec64.pth'


    sam_model = SAM_allinsam.SAM(inp_size=1024, encoder_mode=encoder_mode, loss='iou', device=device)
    sam_model.optimizer = torch.optim.AdamW(sam_model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sam_model.optimizer, 20, eta_min=1.0e-7)


    sam_model = load_checkpoint(sam_model, sam_checkpiont)

    for name, para in sam_model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
        if "prompt_encoder" in name:
            para.requires_grad_(False)
        # if "mask_decoder" in name:
        #     para.requires_grad_(False)

    sam_model = sam_model.cuda()

    print(torch.cuda.memory_allocated() / 1024 / 1024, '******')
    model_total_params = sum(p.numel() for p in sam_model.parameters())
    model_grad_params = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    train_dataset = MoNuSeg_weak_dataset(args, 'train', ssl=True)
    val_dataset = MoNuSeg_weak_dataset(args, 'val', ssl=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_dataloader = DataLoader(val_dataset)

    max_Dice, max_Aji = 0, 0
    total_train_loss = []
    for epoch in range(args.epochs):
        if args.resume != 0:
            epoch += args.resume

    # if (epoch + 1) % 20 == 0:
    #     sam_model = load_checkpoint(sam_model, os.path.join(args.model, str(epoch)+ '_model.pth'))

        os.makedirs(os.path.join(args.result, str(epoch)), exist_ok=True)
        sam_model.train()
        train_loss = 0

        for iter, batch in enumerate(train_dataloader): # batch[0]
            img = batch[0][0]

            img = F.interpolate(img, (args.img_size, args.img_size), mode='bilinear', align_corners=True)
            label = batch[0][1]/255
            point = batch[0][2]
            # clu_label = batch[0][3].squeeze(1)
            # vor_label = batch[0][4].squeeze(1)
            img_name = batch[1][0]

            sam_model.set_input(img, label)#, clu_label, vor_label)
            sam_model.optimize_parameters()

            # bce_loss, offset_gt, offset_loss, iou_loss = backwards
            # bce_local_loss, iou_local_loss = backwards_local
            lr_scheduler.step()

            loss = sam_model.loss_G

            train_loss += loss / len(train_dataloader)

            if (iter + 1) % args.print_fq == 0:
                print('{}/{} epoch, {}/{} batch, train loss: {}'.format(epoch,
                                                                        args.epochs,
                                                                        iter + 1,
                                                                        len(train_dataloader),
                                                                        loss))
            gc.collect()
            torch.cuda.empty_cache()
        total_train_loss.append(train_loss)
        print('{} epoch, mean train loss: {}'.format(epoch, total_train_loss[-1]))
        # print(a)
        # save_checkpoint(os.path.join(args.model, str(epoch) + '_model.pth'), sam_model, epoch)

        if epoch >= args.start_val:
            sam_model.eval()
            mean_dice, mean_iou, mean_aji = 0, 0, 0

            with torch.no_grad():
                for iter, pack in enumerate(val_dataloader):
                    input = pack[0][0]
                    mask = pack[0][1]
                    img_name = pack[1][0]

                    output = split_forward(sam_model, input, mask, args.img_size, device, args.num_hq_token)
                    binary_mask = torch.sigmoid(output[0][0]).detach().cpu().numpy()
                    from skimage import measure
                    binary_map = measure.label(binary_mask> 0.5)


                    if len(np.unique(binary_map)) == 1:
                        dice, iou, aji = 0, 0, 0
                    else:
                        # dice, iou = accuracy_object_level(instance_map, mask[0][0].detach().cpu().numpy())
                        # aji = AJI_fast(mask[0][0].detach().cpu().numpy(), instance_map)
                        dice, iou = accuracy_object_level(binary_map, mask[0][0].detach().cpu().numpy())
                        aji = AJI_fast(mask[0][0].detach().cpu().numpy(), binary_map)

                    mean_dice += dice / len(val_dataloader)  # *len(local_rank))
                    mean_iou += iou / len(val_dataloader)  # len(local_rank))
                    mean_aji += aji / len(val_dataloader)  # *len(local_rank))
                    # print(len(val_dataloader), mean_dice, mean_aji)

                    binary_mask = (binary_mask-np.min(binary_mask))/(np.max(binary_mask)-np.min(binary_mask))
                    binary_mask = Image.fromarray((binary_mask*255).astype(np.uint8))
                    binary_mask.save(os.path.join(args.result, str(epoch), str(img_name) + '_prob.png'))

                    binary_map = mk_colored(binary_map) * 255
                    binary_map = Image.fromarray((binary_map).astype(np.uint8))
                    binary_map.save(os.path.join(args.result, str(epoch), str(img_name) + '_pred.png'))

                    mask = mk_colored(mask[0][0].detach().cpu().numpy()) * 255
                    mask = Image.fromarray((mask).astype(np.uint8))
                    mask.save(os.path.join(args.result, str(epoch), str(img_name) + '_mask.png'))

                    del binary_map, mask, input

                f = open(os.path.join(args.result, str(epoch), "result.txt"), 'w')
                f.write('***test result_mask*** Average- Dice\tIOU\tAJI: '
                        '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_dice, mean_iou, mean_aji))
                f.close()

                if max_Dice < mean_dice:
                    print('save {} epoch!!--Dice: {}'.format(str(epoch), mean_dice))
                    save_checkpoint(os.path.join(args.model, 'Dice_best_model.pth'), sam_model, epoch)
                    max_Dice = mean_dice

                if max_Aji < mean_aji:
                    print('save {} epoch!!--Aji: {}'.format(str(epoch), mean_aji))
                    save_checkpoint(os.path.join(args.model, 'Aji_best_model.pth'), sam_model, epoch)
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

    sam_model = SAM_allinsam.SAM(inp_size=1024, encoder_mode=encoder_mode, loss='iou', device=device)
    sam_model = sam_model.cuda()

    # sam_checkpoint = torch.load(os.path.join(args.model, 'Aji_best_model.pth'))
    # sam_model.load_state_dict(sam_checkpoint, strict=False)
    sam_model = load_checkpoint(sam_model, os.path.join(args.model, 'Aji_best_model.pth'))
    # sam_model = load_checkpoint(sam_model, os.path.join(args.model, 'Dice_best_model.pth'))

    test_dataseet = MoNuSeg_weak_dataset(args, 'test')
    test_dataloader = DataLoader(test_dataseet)

    os.makedirs(os.path.join(args.result,'test'), exist_ok=True)
    sam_model.eval()
    mean_dice, mean_iou, mean_aji = 0, 0, 0
    # if torch.distributed.get_rank() == 0:

    with torch.no_grad():
        for iter, pack in enumerate(test_dataloader):
            input = pack[0][0]
            mask = pack[0][1]
            img_name = pack[1][0]

            output = split_forward(sam_model, input, mask, args.img_size, device, args.num_hq_token)
            binary_mask = torch.sigmoid(output[0][0]).detach().cpu().numpy()
            from skimage import measure
            binary_map = measure.label(binary_mask > 0.5)

            if len(np.unique(binary_map)) == 1:
                dice, iou, aji = 0, 0, 0
            else:
                # dice, iou = accuracy_object_level(instance_map, mask[0][0].detach().cpu().numpy())
                # aji = AJI_fast(mask[0][0].detach().cpu().numpy(), instance_map)
                dice, iou = accuracy_object_level(binary_map, mask[0][0].detach().cpu().numpy())
                aji = AJI_fast(mask[0][0].detach().cpu().numpy(), binary_map)

            mean_dice += dice / len(test_dataloader)  # *len(local_rank))
            mean_iou += iou / len(test_dataloader)  # len(local_rank))
            mean_aji += aji / len(test_dataloader)  # *len(local_rank))
            # print(len(val_dataloader), mean_dice, mean_aji)

            binary_mask = (binary_mask - np.min(binary_mask)) / (np.max(binary_mask) - np.min(binary_mask))
            binary_mask = Image.fromarray((binary_mask * 255).astype(np.uint8))
            binary_mask.save(os.path.join(args.result, 'test', str(img_name) + '_prob.png'))

            binary_map = mk_colored(binary_map) * 255
            binary_map = Image.fromarray((binary_map).astype(np.uint8))
            binary_map.save(os.path.join(args.result, 'test', str(img_name) + '_pred.png'))

            mask = mk_colored(mask[0][0].detach().cpu().numpy()) * 255
            mask = Image.fromarray((mask).astype(np.uint8))
            mask.save(os.path.join(args.result, 'test', str(img_name) + '_mask.png'))



    f = open(os.path.join(args.result, 'test', "result.txt"), 'w')
    f.write('***test result_mask*** Average- Dice\tIOU\tAJI: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_dice, mean_iou, mean_aji))
    f.close()
    #     # if min_loss > val_losses:
    #     #     print('save {} epoch!!--loss: {}'.format(str(epoch), val_losses))
    #     #     save_checkpoint(os.path.join(args.model, 'loss_best_model.pth'), model, epoch)
    #     #     min_loss = val_losses

    print('test result: Average- Dice\tIOU\tAJI: '
                 '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_dice, mean_iou, mean_aji))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--resume', default=0, type=int, help='')
    parser.add_argument('--start_val', default=30, type=int)
    parser.add_argument('--plt', action='store_true')
    parser.add_argument('--sup', action='store_true')

    # parser.add_argument('--data',default='/media/NAS/nas_70/open_dataset/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg',help='path to dataset')
    # parser.add_argument('--data', default='/media/NAS/nas_70/open_dataset/MoNuSeg/MoNuSeg_shift8/via instance learning data_for_train/MoNuSeg_shift8', help='path to dataset')

    parser.add_argument('--data',default='/media/NAS/nas_70/open_dataset/CPM/CPM 17/via instance learning data_for_train/CPM 17',help='path to dataset')
    # parser.add_argument('--data', default='/media/NAS/nas_70/open_dataset/CPM/CPM 17_shift6/via instance learning data_for_train/CPM 17_shift6', help='path to dataset')

    # parser.add_argument('--data', default='/media/NAS/nas_70/open_dataset/CoNSeP/CoNSeP/via instance learning data_for_train/CoNSeP', help='path to dataset')
    # parser.add_argument('--data', default='/media/NAS/nas_70/open_dataset/CoNSeP/CoNSeP_shift4/via instance learning data_for_train/CoNSeP_shift4', help='path to dataset')

    # parser.add_argument('--data', default='/media/NAS/nas_32/siwoo/TNBC/TNBC/via instance learning data_for_train/TNBC', help='path to dataset')
    # parser.add_argument('--data', default='/media/NAS/nas_32/siwoo/TNBC/TNBC_shift4/via instance learning data_for_train/TNBC_shift4', help='path to dataset')

    parser.add_argument('--img_size', default=1024, help='')
    parser.add_argument('--num_hq_token', default=8, type=int, help='')
    parser.add_argument('--epochs', default=100, type=int, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='')
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--model_type', default='vit_h', help='')
    # parser.add_argument('--sam_checkpoint', default='/media/NAS/nas_187/siwoo/2023/result/transformer_freeze_new_h2_MO/model/Aji_best_model.pth')
    parser.add_argument('--print_fq', default=15, type=int, help='')
    parser.add_argument('--ord_th', default=0.5, type=float)

    # parser.add_argument('--result', default='/media/NAS/nas_187/siwoo/2023/result/MO_shift4_fs2/img', help='')
    # parser.add_argument('--model', default='/media/NAS/nas_187/siwoo/2023/result/MO_shift_4_fs2/model', help='')
    parser.add_argument('--result', default='/media/NAS/nas_187/siwoo/2023/result/MO_allinsam/img', help='')
    parser.add_argument('--model', default='/media/NAS/nas_187/siwoo/2023/result/MO_allinsam/model', help='')

    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()



    if args.model != None:
        os.makedirs(args.model, exist_ok=True)
    if args.result != None:
        os.makedirs(args.result, exist_ok=True)

    print('=' * 40)
    print(' ' * 14 + 'Arguments')
    for arg in sorted(vars(args)):
        print(arg + ':', getattr(args, arg))
    print('=' * 40)


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # main(args)
    test(args, device)














