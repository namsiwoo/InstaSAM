import torch
import os
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from datasets.MoNuSeg_dataset import MoNuSeg_weak_dataset
from skimage import measure

import models

from models.mmseg.models.sam import PromptEncoder

# from segment_anything import sam_model_registry
# from utils.transforms import ResizeLongestSide
from utils.utils import accuracy_object_level, AJI_fast, save_checkpoint, load_checkpoint, mk_colored

from torch.nn.functional import threshold, normalize

def extract_center_points(mask):
    point_mask = np.zeros_like(mask.numpy())
    for index in (np.unique(mask.numpy())[1:]):
        coor = np.where(mask.numpy() == index)
        x = np.array(coor[1])
        y = np.array(coor[0])
        point_mask[round(np.mean(y)), round(np.mean(x))] = 1
    return point_mask

def make_point_prompt(point_mask):
    point_mask = measure.label(point_mask)
    point_coords = []
    for index in np.unique(point_mask)[1:]:
        coor = np.where(point_mask == index)
        x = np.array(coor[1])
        y = np.array(coor[0])
        point_coords.append([np.mean(x), np.mean(y)])
    point_labels = np.ones(len(point_coords), dtype=int)
    return point_coords, point_labels

    # point_coords = []
    # for index in (np.unique(mask.numpy())[1:]):
    #     coor = np.where(mask.numpy() == index)
    #     x = np.array(coor[1])
    #     y = np.array(coor[0])
    #     point_coords.append([np.mean(x), np.mean(y)])
    # point_labels = np.ones(len(point_coords), dtype=int)
    # return point_coords, point_labels

def apply_coords_torch(coords: torch.Tensor, original_size, target_size) -> torch.Tensor:
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    from copy import deepcopy
    """
    Expects a torch tensor with length 2 in the last dimension. Requires the
    original image size in (H, W) format.
    """

    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(
        original_size[0], original_size[1], target_size
    )
    coords = deepcopy(coords).to(torch.float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords

def add_margin(pil_img, top, right, bottom, left, color):
    """pading on PIL image"""
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def norm(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def torch_norm(img):
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))

def split_forward(sam_model, input, mask, sam_input_size, device):
    size = 224
    overlap = 80
    point_mask = extract_center_points(mask[0][0])

    b, c, h0, w0 = input.size()
    # print(input.shape)
    # print(np.unique(input.numpy()))
    # plt.imshow(input[0].numpy().transpose(1, 2, 0))
    # plt.savefig('/media/NAS/nas_187/siwoo/2023/result/finetune/img/'+str(0)+'/'+'input_ex.png')

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

    # plt.imshow(input[0].numpy().transpose(1, 2, 0))
    # plt.savefig('/media/NAS/nas_187/siwoo/2023/result/finetune/img/'+str(0)+'/'+str(1)+'ex.png')

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

            point_mask_patch = point_mask[i:r_end, j:c_end]
            # mask_patch = mask[:, :, i:r_end, j:c_end]
            point_coords_ori, point_labels = make_point_prompt(point_mask_patch) # n, 2
            point_coords = apply_coords_torch(torch.from_numpy(np.array(point_coords_ori)).unsqueeze(0),(224, 224), sam_model.image_encoder.img_size)
            point_coords = torch.as_tensor(point_coords, dtype=torch.float, device=device)
            point_labels = torch.as_tensor(point_labels, dtype=torch.int, device=device).unsqueeze(0)
                # point_coords, point_labels = point_coords[:, None, :], point_labels[:, None]


                # plt.imshow(input_patch[0].numpy().transpose(1, 2, 0))
                # plt.savefig('/media/NAS/nas_187/siwoo/2023/result/finetune/img/' + str(0) + '/' + str(333) + 'ex.png')

                # input_patch = transform.apply_image(input_patch[0].numpy().transpose(1, 2, 0))
                # input_patch = torch.as_tensor(input_patch, device=device)
                # input_patch = input_patch.permute(2, 0, 1).contiguous()[None, :, :, :]
                # input_patch = preprocess(input_patch, device)

                # input_patch = transform.apply_image(input_patch[0])
                # input_patch = preprocess(input_patch, device)

            with torch.no_grad():
                low_res_masks, upscaled_masks = sam_model.infer(input_patch.to(device), (point_coords, point_labels))
                # low_res_masks, upscaled_masks = sam_model.infer(input_patch.to(device), (point_coords, point_labels))
                B, C, H, W = upscaled_masks.shape
                # print(upscaled_masks.shape)

                # for K in range(B):
                #     print(point_coords[K], upscaled_masks.shape)
                #     plt.clf()
                #     plt.subplot(1, 3, 1)
                #     plt.imshow(np.transpose(input_patch[0].numpy(), (1, 2, 0)))
                #     plt.subplot(1, 3, 2)
                #     plt.imshow(mask_patch[0][0])
                #     plt.subplot(1, 3, 3)
                #     plt.imshow(upscaled_masks[K,0].detach().cpu().numpy())
                #     plt.colorbar()
                #     plt.savefig('/media/NAS/nas_187/siwoo/2023/result/finetune_ori_ins/img/0/'+str(K)+'_ex.png')
                #
                #     pred_masks[torch.sigmoid(upscaled_masks[K,0])> 0.5] = K+c*20+1

                    # pred_masks = torch.sigmoid(upscaled_masks[K,0])
            # image_embedding = sam_model.image_encoder(input_patch.to(device))
            #
            # sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            #     points=None,
            #     boxes=None,
            #     masks=None,
            # )
            #
            # low_res_masks, _ = sam_model.mask_decoder(
            #     image_embeddings=image_embedding,
            #     image_pe=sam_model.prompt_encoder.get_dense_pe(),
            #     sparse_prompt_embeddings=sparse_embeddings,
            #     dense_prompt_embeddings=dense_embeddings,
            #     multimask_output=False,
            # )
            # upscaled_masks = sam_model.postprocess_masks(low_res_masks, (1024, 1024), (size, size)).to(device)
            # binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(device)

        # print(torch.unique(pred_masks))
            output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = upscaled_masks[:, :, ind1_s - i:ind1_e - i,
                                                     ind2_s - j:ind2_e - j]

    output = output[:, :, :h0, :w0].to(device)
    return output


def main(args, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # sam_model = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    # sam_model = sam_model.to(device)


    #adapter
    encoder_mode = {'name': 'sam', 'img_size': args.img_size, 'mlp_ratio': 4, 'patch_size': 16, 'qkv_bias': True, 'use_rel_pos': True, 'window_size': 14, 'out_chans': 256, 'scale_factor': 32, 'input_type': 'fft',
                    'freq_nums': 0.25, 'prompt_type': 'highpass', 'prompt_embed_dim': 256, 'tuning_stage': 1234, 'handcrafted_tune': True, 'embedding_tune': True, 'adaptor': 'adaptor', 'embed_dim': 1280,
                    'depth': 32, 'num_heads': 16, 'global_attn_indexes': [7, 15, 23, 31]}

    sam_model = models.sam.SAM(inp_size=1024, encoder_mode=encoder_mode, loss='iou', device=device)
    sam_model.optimizer = torch.optim.AdamW(sam_model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sam_model.optimizer, 20, eta_min=1.0e-7)

    sam_model = sam_model.cuda()
    sam_model = torch.nn.parallel.DistributedDataParallel(
        sam_model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    sam_model = sam_model.module

    # sam_checkpoint = torch.load(args.sam_checkpoint)
    # sam_model.load_state_dict(sam_checkpoint, strict=False)

    sam_model = load_checkpoint(sam_model, args.sam_checkpoint)

    for name, para in sam_model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
        if "prompt_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)

    # prompt_encoder = PromptEncoder(
    #     embed_dim=encoder_mode['prompt_embed_dim'],
    #     image_embedding_size=(1024//encoder_mode['patch_size'],1024//encoder_mode['patch_size']),
    #     input_image_size=(1024,1024),
    #     mask_in_chans=16
    # )
    # prompt_encoder.load_state_dict(sam_checkpoint, strict=False)

    if local_rank == 0:
        print(torch.cuda.memory_allocated() / 1024 / 1024, '******')
        model_total_params = sum(p.numel() for p in sam_model.parameters())
        model_grad_params = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    # optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=0.01) #defalut 0.0005
    # loss_fn = torch.nn.MSELoss().to(device)


    train_dataset = MoNuSeg_weak_dataset(args, 'train')
    val_dataset = MoNuSeg_weak_dataset(args, 'val')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=torch.utils.data.distributed.DistributedSampler(train_dataset))
    val_dataloader = DataLoader(val_dataset, sampler=torch.utils.data.distributed.DistributedSampler(val_dataset))


    # loss_fn = torch.nn.NLLLoss(ignore_index=2).to(device)

    max_Dice, max_Aji = 0, 0
    total_train_loss = []
    for epoch in range(args.epochs):
        os.makedirs(os.path.join(args.result, str(epoch)), exist_ok=True)
        sam_model.train()
        train_dataloader.sampler.set_epoch(epoch)
        train_loss = 0

        loss_list = []
        for iter, batch in enumerate(train_dataloader): # batch[0]
            # batch[0] = batch[0].cuda()

            for i in range(len(batch[0])):
                batch[0][i] = batch[0][i].cuda()

            # img = transform.apply_image(pack[0][0])
            img = batch[0][0]
            img = F.interpolate(img, (args.img_size, args.img_size), mode='bilinear', align_corners=True)
            point = batch[0][1]/255


            label = batch[0][4]
            if len(label.shape) == 3:
                label = label.unsqueeze(1)

            # label = label.transpose(0, 1)
            # label = torch.argmax(label, dim=0).unsqueeze(0)
            label[label>0] = 1.

            # plt.imshow(point[0][0].detach().cpu().numpy())
            # plt.savefig('/media/NAS/nas_187/siwoo/2023/ex.png')

            point_coords2, point_labels = batch[0][2], batch[0][3] # 1, 20, 2 // 1, 20
            point_coords = apply_coords_torch(point_coords2, (224, 224), args.img_size)

            sam_model.set_input(img, label)
            low_res_masks, upscaled_masks = sam_model.optimize_parameters() # (point_coords, point_labels)
            lr_scheduler.step()

            # upscaled_masks = normalize(upscaled_masks).to(device)

            # ????
            batch_loss = [torch.zeros_like(sam_model.loss_G) for _ in range(dist.get_world_size())]
            dist.all_gather(batch_loss, sam_model.loss_G)
            loss_list.extend(batch_loss)

            # # loss = loss_fn(upscaled_masks, label.float())
            # print(loss)
            # print('----')
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            loss = np.array([i.item() for i in loss_list])

            # train_loss += np.mean(loss) / len(train_dataloader)
            train_loss += np.sum(loss) / len(train_dataloader)

            if torch.distributed.get_rank() == 0 and (iter + 1) % args.print_fq == 0:
                print('{}/{} epoch, {}/{} batch, train loss: {}'.format(epoch, args.epochs, iter + 1, len(train_dataloader), np.mean(loss)))

                # print(np.unique(upscaled_masks[0][0].detach().cpu()), img.shape)
                plt.clf()
                plt.subplot(1, 3, 1)
                img1 = norm(img.detach().cpu().numpy()[0].transpose(1, 2, 0))
                plt.imshow(img1)
                plt.subplot(1, 3, 2)
                plt.imshow(label.detach().cpu().numpy()[0,0])
                plt.subplot(1, 3, 3)
                plt.imshow(upscaled_masks.detach().cpu().numpy()[0,0])
                plt.colorbar()

                # plt.subplot(2, 3, 4)
                # img2 = norm(img.detach().cpu().numpy()[1].transpose(1, 2, 0))
                # plt.imshow(img2)
                # plt.subplot(2, 3, 5)
                # plt.imshow(label.detach().cpu().numpy()[1,0])
                # plt.subplot(2, 3, 6)
                # plt.imshow(upscaled_masks.detach().cpu().numpy()[1,0])
                # plt.colorbar()
                plt.savefig(os.path.join(args.result,str(epoch), str(iter) + 'ex.png'))

        total_train_loss.append(train_loss)
        if torch.distributed.get_rank() == 0:
            print('{} epoch, mean train loss: {}'.format(epoch, total_train_loss[-1]))
        save_checkpoint(os.path.join(args.model, str(epoch) + '_model.pth'), sam_model, epoch)

        sam_model.eval()
        mean_dice, mean_iou, mean_aji = torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda()
        # if torch.distributed.get_rank() == 0:
        with torch.no_grad():
            for iter, pack in enumerate(val_dataloader):
                input = pack[0][0]
                mask = pack[0][1]
                img_name = pack[1][0]

                output = split_forward(sam_model, input, mask, args.img_size, device)
                binary_mask = torch.sigmoid(output).detach().cpu().numpy()>0.5
                # binary_mask = output.detach().cpu().numpy()


                if len(np.unique(binary_mask)) == 1:
                    dice, iou, aji = 0, 0, 0
                else:
                    dice, iou = accuracy_object_level(binary_mask[0][0], mask[0][0].detach().cpu().numpy())
                    aji = AJI_fast(mask[0][0].detach().cpu().numpy(), binary_mask[0][0])

                mean_dice += dice / (len(val_dataloader)*3)#*len(local_rank))
                mean_iou += iou / (len(val_dataloader)*3)#len(local_rank))
                mean_aji += aji / (len(val_dataloader)*3)#*len(local_rank))
                # print(len(val_dataloader), mean_dice, mean_aji)

                dist.reduce(mean_dice, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(mean_iou, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(mean_aji, dst=0, op=dist.ReduceOp.SUM)

                binary_mask = measure.label(binary_mask[0][0], connectivity=2)

                pred = mk_colored(binary_mask) * 255
                pred = Image.fromarray((pred).astype(np.uint8))
                pred.save(os.path.join(args.result, str(epoch), str(img_name) + '_pred.png'))

                mask = mk_colored(mask[0][0].detach().cpu().numpy()) * 255
                mask = Image.fromarray((mask).astype(np.uint8))
                mask.save(os.path.join(args.result, str(epoch), str(img_name) + '_mask.png'))



        if torch.distributed.get_rank() == 0:
            f = open(os.path.join(args.result, str(epoch), "result.txt"), 'w')
            f.write('***test result_mask*** Average- Dice\tIOU\tAJI: '
                    '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_dice[0], mean_iou[0], mean_aji[0]))
            f.close()
            #     # if min_loss > val_losses:
            #     #     print('save {} epoch!!--loss: {}'.format(str(epoch), val_losses))
            #     #     save_checkpoint(os.path.join(args.model, 'loss_best_model.pth'), model, epoch)
            #     #     min_loss = val_losses
            if max_Dice < mean_dice[0]:
                print('save {} epoch!!--Dice: {}'.format(str(epoch), mean_dice[0]))
                save_checkpoint(os.path.join(args.model, 'Dice_best_model.pth'), sam_model, epoch)
                max_Dice = mean_dice[0]

            if max_Aji < mean_aji[0]:
                print('save {} epoch!!--Aji: {}'.format(str(epoch), mean_aji[0]))
                save_checkpoint(os.path.join(args.model, 'Aji_best_model.pth'), sam_model, epoch)
                max_Aji = mean_aji[0]

            print(epoch, ': Average- Dice\tIOU\tAJI: '
                         '\t\t{:.4f}\t{:.4f}\t{:.4f} (b Dice: {}, b Aji: {})'.format(mean_dice[0], mean_iou[0],
                                                                                mean_aji[0], max_Dice, max_Aji))


def test(args, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    encoder_mode = {'name': 'sam', 'img_size': args.img_size, 'mlp_ratio': 4, 'patch_size': 16, 'qkv_bias': True, 'use_rel_pos': True, 'window_size': 14, 'out_chans': 256, 'scale_factor': 32, 'input_type': 'fft',
                    'freq_nums': 0.25, 'prompt_type': 'highpass', 'prompt_embed_dim': 256, 'tuning_stage': 1234, 'handcrafted_tune': True, 'embedding_tune': True, 'adaptor': 'adaptor', 'embed_dim': 1280,
                    'depth': 32, 'num_heads': 16, 'global_attn_indexes': [7, 15, 23, 31]}
    sam_model = models.sam.SAM(inp_size=1024, encoder_mode=encoder_mode, loss='bce', device=device)

    sam_model = sam_model.cuda()
    sam_model = torch.nn.parallel.DistributedDataParallel(
        sam_model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    sam_model = sam_model.module

    # sam_checkpoint = torch.load(os.path.join(args.model, 'Aji_best_model.pth'))
    # sam_model.load_state_dict(sam_checkpoint, strict=False)
    sam_model = load_checkpoint(sam_model, os.path.join(args.model, 'Aji_best_model.pth'))

    test_dataseet = MoNuSeg_weak_dataset(args, 'test')
    test_dataloader = DataLoader(test_dataseet, sampler=torch.utils.data.distributed.DistributedSampler(test_dataseet))

    sam_model.eval()
    mean_dice, mean_iou, mean_aji = torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda()
    # if torch.distributed.get_rank() == 0:
    os.makedirs(os.path.join(args.result,'test'), exist_ok=True)
    with torch.no_grad():
        for iter, pack in enumerate(test_dataloader):
            input = pack[0][0]
            mask = pack[0][1].to(device)
            img_name = pack[1][0]

            output = split_forward(sam_model, input, mask, args.img_size, device)
            binary_mask = norm(output.detach().cpu().numpy()) > 0.5

            if len(np.unique(binary_mask)) == 1:
                dice, iou, aji = 0, 0, 0
            else:
                dice, iou = accuracy_object_level(binary_mask[0][0], mask[0][0].detach().cpu().numpy())
                aji = AJI_fast(mask[0][0].detach().cpu().numpy(), binary_mask[0][0])

            mean_dice += dice / (len(test_dataloader) * 1)  # *len(local_rank))
            mean_iou += iou / (len(test_dataloader) * 1)  # len(local_rank))
            mean_aji += aji / (len(test_dataloader) * 1)

            # print(len(val_dataloader), mean_dice, mean_aji)

            # dist.reduce(mean_dice, dst=0, op=dist.ReduceOp.SUM)
            # dist.reduce(mean_iou, dst=0, op=dist.ReduceOp.SUM)
            # dist.reduce(mean_aji, dst=0, op=dist.ReduceOp.SUM)

            binary_mask = measure.label(binary_mask[0][0], connectivity=2)

            pred = mk_colored(binary_mask) * 255
            pred = Image.fromarray((pred).astype(np.uint8))
            pred.save(os.path.join(args.result, 'test', str(img_name) + '_pred.png'))

            mask = mk_colored(mask[0][0].detach().cpu().numpy()) * 255
            mask = Image.fromarray((mask).astype(np.uint8))
            mask.save(os.path.join(args.result, 'test', str(img_name) + '_mask.png'))


    if torch.distributed.get_rank() == 0:
        f = open(os.path.join(args.result, 'test', "result.txt"), 'w')
        f.write('***test result_mask*** Average- Dice\tIOU\tAJI: '
                '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_dice[0], mean_iou[0], mean_aji[0]))
        f.close()
        #     # if min_loss > val_losses:
        #     #     print('save {} epoch!!--loss: {}'.format(str(epoch), val_losses))
        #     #     save_checkpoint(os.path.join(args.model, 'loss_best_model.pth'), model, epoch)
        #     #     min_loss = val_losses

        if torch.distributed.get_rank() == 0:
            print('test result: Average- Dice\tIOU\tAJI: '
                         '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_dice[0], mean_iou[0], mean_aji[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data',default='/media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg',help='path to dataset')
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument('--img_size', default=1024, help='')
    parser.add_argument('--epochs', default=100, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='')
    parser.add_argument('--model_type', default='vit_h', help='')
    parser.add_argument('--sam_checkpoint', default='sam_vit_h_4b8939.pth')
    parser.add_argument('--print_fq', default=5, help='')

    parser.add_argument('--result', default='/media/NAS/nas_187/siwoo/2023/result/sam_adapter_iou_nopoint/img', help='')
    parser.add_argument('--model', default='/media/NAS/nas_187/siwoo/2023/result/sam_adapter_iou_nopoint/model', help='')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')
    args = parser.parse_args()

    if args.model != None:
        os.makedirs(args.model, exist_ok=True)
    if args.result != None:
        os.makedirs(args.result, exist_ok=True)

    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    dist.barrier()
    device = torch.device("cuda", local_rank)

    # if torch.distributed.get_rank() == 0:
    #     print('=' * 40)
    #     print(' ' * 14 + 'Arguments')
    #     for arg in sorted(vars(args)):
    #         print(arg + ':', getattr(args, arg))
    #     print('=' * 40)

    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    main(args, device)
    test(args, device)

    # CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 3 train_sam_adaptor.py --batch_size 1