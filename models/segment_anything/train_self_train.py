import torch
import os
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import skimage.morphology as ski_morph
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Dataset.MoNuSeg_dataset import MoNuSeg_weak_dataset
from scipy.ndimage.morphology import binary_fill_holes
from skimage import measure
from segment_anything import sam_model_registry
from utils.transforms import ResizeLongestSide
from utils.utils import accuracy_object_level, AJI_fast, save_checkpoint, load_checkpoint, mk_colored
from torch.nn.functional import threshold, normalize
from utils.utils import accuracy_object_level, AJI_fast, save_checkpoint, load_checkpoint, mk_colored
from utils.loss import IOU_loss
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
            point_coords_ori, point_labels_ori = make_point_prompt(point_mask_patch) # n, 2

            pred_masks = torch.zeros((1, 224, 224)).to(device) + 0.5
            for c in range(0, len(point_labels_ori)//20):
                if (c+1)*20 > len(point_labels_ori):
                    point_coords = apply_coords_torch(torch.from_numpy(np.array(point_coords_ori)[c*20:]).unsqueeze(0),(224, 224), sam_model.image_encoder.img_size)
                    point_labels = point_labels_ori[c*20:]
                else:
                    point_coords = apply_coords_torch(torch.from_numpy(np.array(point_coords_ori)[c*20:(c+1)*20]).unsqueeze(0),(224, 224), sam_model.image_encoder.img_size)
                    point_labels = point_labels_ori[c*20:(c+1)*20]

                point_coords = torch.as_tensor(point_coords.squeeze(0).unsqueeze(1), dtype=torch.float, device=device)
                point_labels = torch.as_tensor(point_labels, dtype=torch.int, device=device).unsqueeze(1)
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
                    image_embedding = sam_model.image_encoder(input_patch.to(device))

                    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                        points=(point_coords, point_labels),
                        boxes=None,
                        masks=None,
                    )

                    low_res_masks, _ = sam_model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    upscaled_masks = sam_model.postprocess_masks(low_res_masks, (1024, 1024), (size, size)).to(device)

                    # low_res_masks, upscaled_masks = sam_model.infer(input_patch.to(device), (point_coords, point_labels))
                    # print(upscaled_masks.shape)

                    pred_masks = torch.cat([pred_masks, upscaled_masks.squeeze(1)])
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
            pred_masks = torch.argmax(pred_masks, dim=0)
            output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = pred_masks[ind1_s - i:ind1_e - i,
                                                         ind2_s - j:ind2_e - j]


    output = output[:, :, :h0, :w0].to(device)
    return output

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    sam_model = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam_model = sam_model.to(device)
    for name, para in sam_model.named_parameters():
        if "image_encoder" in name or 'prompt_encoder' in name:
            para.requires_grad_(False)
    print(torch.cuda.memory_allocated() / 1024 / 1024, '******')
    model_total_params = sum(p.numel() for p in sam_model.parameters())
    model_grad_params = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    train_dataset = MoNuSeg_weak_dataset(args, 'train')
    val_dataset = MoNuSeg_weak_dataset(args, 'val')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset)

    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=1.0e-7)


    criterionBCE = torch.nn.BCEWithLogitsLoss().to(device)
    criterionIOU = IOU_loss().to(device)
    # loss_fn = torch.nn.NLLLoss(ignore_index=2).to(device)

    max_Dice, max_Aji = 0, 0
    total_train_loss = []
    for epoch in range(args.epochs):
        os.makedirs(os.path.join(args.result, str(epoch)), exist_ok=True)
        sam_model.train()
        train_loss = 0
        for iter, pack in enumerate(train_dataloader):
            # img = transform.apply_image(pack[0][0])
            img = pack[0][0].to(device)
            img = F.interpolate(img, (sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), mode='bilinear', align_corners=True)
            point = pack[0][1]/255

            label = pack[0][4].to(device)
            label = label.transpose(0, 1)
            point_coords2, point_labels = pack[0][2].to(device), pack[0][3].to(device) # 1, 20, 2 // 1, 20
            point_coords = apply_coords_torch(point_coords2, (224, 224), sam_model.image_encoder.img_size)
            point_coords, point_labels = point_coords.squeeze(0).unsqueeze(1), point_labels.squeeze(0).unsqueeze(1)


            # image embedding
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(img)

            with torch.no_grad():
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None,
                    masks=None,
                )

            low_res_masks, _ = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            upscaled_masks = sam_model.postprocess_masks(low_res_masks, (1024, 1024), (224, 224)).to(device)
            # upscaled_masks = normalize(upscaled_masks).to(device)

            # print(np.unique(upscaled_masks[0][0].detach().cpu()), img.shape)
            # plt.subplot(1, 3, 1)
            # plt.imshow(img.detach().cpu().numpy()[0].transpose(1, 2, 0))
            # plt.subplot(1, 3, 2)
            # plt.imshow(label.detach().cpu().numpy()[0, 0])
            # plt.subplot(1, 3, 3)
            # plt.imshow(upscaled_masks[0][0].detach().cpu().numpy())
            # plt.savefig('/media/NAS/nas_187/siwoo/2023/result/finetune/img/'+str(epoch)+'/'+str(iter)+'ex.png')

            # print(np.unique(label.detach().cpu()), np.unique(upscaled_masks.detach().cpu()))

            loss = criterionBCE(upscaled_masks, label.float()) + criterionIOU(upscaled_masks, label.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss += loss.item() / len(train_dataloader)

            if (iter + 1) % args.print_fq == 0:
                print('{}/{} epoch, {}/{} batch, train loss: {}'.format(epoch, args.epochs, iter + 1, len(train_dataloader), loss.item()))
                plt.subplot(2, 3, 1)
                img = norm(img.detach().cpu().numpy()[0].transpose(1, 2, 0))
                if len(point_coords[0]) > 5:
                    num = 5
                else:
                    num = len(point_coords[0])
                for k in range(num):
                    for i in range(15):
                        for j in range(15):
                            img[round(point_coords[0][k][1].detach().cpu().numpy()-i), round(point_coords[0][k][0].detach().cpu().numpy()-j), :] = 0
                plt.imshow(img)
                plt.subplot(2, 3, 4)
                # plt.imshow(np.argmax(label.detach().cpu().numpy(), 0)[0])
                plt.imshow(point.detach().cpu().numpy()[0][0])

                for i in range(2):
                    plt.subplot(2, 3, 2+i)
                    if i == 0:
                        # aaaa = norm(upscaled_masks[i][0].detach().cpu().numpy())
                        aaaa = torch.sigmoid(upscaled_masks[i][0]).detach().cpu().numpy()
                        x = int(point_coords2[0][i][1].detach().cpu().numpy())
                        y = int(point_coords2[0][i][0].detach().cpu().numpy())
                        for k in range(-3, 0):
                            for j in range(-3, 0):
                                aaaa[x+k, y+j] = -1
                    else:
                        aaaa = np.argmax(label.detach().cpu().numpy(), 0)[0]
                        aaaa[aaaa>0] = -1
                        aaaa[label[0, 0].detach().cpu().numpy()>0] = 1
                    plt.imshow(aaaa)
                    plt.colorbar()
                for i in range(2):
                    plt.subplot(2, 3, 5 + i)
                    if i == 0:
                        aaaa = norm(upscaled_masks[2+i][0].detach().cpu().numpy())
                        for k in range(-3, 0):
                            for j in range(-3, 0):
                                aaaa[int(point_coords2[0][1+i][1].detach().cpu().numpy()+k), int(point_coords2[0][2+i][0].detach().cpu().numpy())+j] = -1
                    else:
                        aaaa = np.argmax(label.detach().cpu().numpy(), 0)[0]
                        aaaa[aaaa>0] = -1
                        aaaa[label[1, 0].detach().cpu().numpy()>0] = 1
                    plt.imshow(aaaa)
                    plt.colorbar()
                plt.savefig(os.path.join(args.result,str(epoch), str(iter) + 'ex.png'))

        total_train_loss.append(train_loss)
        print('---------------------------------------------------------------')
        print('{} epoch, mean train loss: {}'.format(epoch, total_train_loss[-1]))
        save_checkpoint(os.path.join(args.model, str(epoch) + '_model.pth'), sam_model, epoch)

        sam_model.eval()
        mean_dice, mean_iou, mean_aji = 0, 0, 0
        with torch.no_grad():
            for iter, pack in enumerate(val_dataloader):
                input = pack[0][0]
                mask = pack[0][1]
                img_name = pack[1][0]

                output = split_forward(sam_model, input, mask, sam_model.image_encoder.img_size, device)
                # binary_mask = normalize(threshold(output, 0.0, 0)).detach().cpu().numpy()
                # binary_mask = torch.sigmoid(output).detach().cpu().numpy()>0.5 # maybe it is right
                binary_mask = output.detach().cpu().numpy()


                if len(np.unique(binary_mask)) == 1:
                    dice, iou, aji = 0, 0, 0
                else:
                    dice, iou = accuracy_object_level(binary_mask[0][0], mask[0][0].detach().cpu().numpy())
                    aji = AJI_fast(mask[0][0].detach().cpu().numpy(), binary_mask[0][0])

                mean_dice += dice / len(val_dataloader)
                mean_iou += iou / len(val_dataloader)
                mean_aji += aji / len(val_dataloader)

                pred = mk_colored(binary_mask[0][0]) * 255
                pred = Image.fromarray((pred).astype(np.uint8))
                pred.save(os.path.join(args.result, str(epoch), str(img_name) + '_pred.png'))

                mask = mk_colored(mask[0][0].detach().cpu().numpy()) * 255
                mask = Image.fromarray((mask).astype(np.uint8))
                mask.save(os.path.join(args.result, str(epoch), str(img_name) + '_mask.png'))


        f = open(os.path.join(args.result, str(epoch), "result.txt"), 'w')
        f.write('***test result_mask*** Average- Dice\tIOU\tAJI: '
                '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_dice, mean_iou, mean_aji))
        f.close()
        #     # if min_loss > val_losses:
        #     #     print('save {} epoch!!--loss: {}'.format(str(epoch), val_losses))
        #     #     save_checkpoint(os.path.join(args.model, 'loss_best_model.pth'), model, epoch)
        #     #     min_loss = val_losses
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


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    sam_model = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam_model = sam_model.to(device)

    test_dataseet = MoNuSeg_weak_dataset(args, 'test')
    test_dataloader = DataLoader(test_dataseet)

    transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    sam_model.eval()
    mean_dice, mean_iou, mean_aji = 0, 0, 0
    os.makedirs(os.path.join(args.result, 'test'), exist_ok=True)
    with torch.no_grad():
        for iter, pack in enumerate(test_dataloader):
            input = pack[0][0]
            mask = pack[0][1].to(device)
            img_name = pack[1]

            output = split_forward(sam_model, input, transform, device)
            binary_mask = normalize(threshold(output, 0.0, 0)).to(device)

            dice, iou = accuracy_object_level(binary_mask[0][0], mask[0][0].detach().cpu().numpy())
            aji = AJI_fast(mask[0][0].detach().cpu().numpy(), binary_mask[0][0])

            mean_dice += dice / len(test_dataloader)
            mean_iou += iou / len(test_dataloader)
            mean_aji += aji / len(test_dataloader)

            pred = mk_colored(binary_mask[0][0]) * 255
            pred = Image.fromarray((pred).astype(np.uint8))
            pred.save(os.path.join(args.result, 'test', str(img_name) + '_pred.png'))

            mask = mk_colored(mask) * 255
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
    parser.add_argument('--data',default='/media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg',help='path to dataset')
    parser.add_argument('--epochs', default=100, help='')
    parser.add_argument('--batch_size', default=4, type=int, help='')
    parser.add_argument('--model_type', default='vit_h', help='')
    parser.add_argument('--sam_checkpoint', default='sam_vit_h_4b8939.pth')
    parser.add_argument('--print_fq', default=40, help='')

    parser.add_argument('--result', default='/media/NAS/nas_187/siwoo/2023/result/finetune_ori_ins/img', help='')
    parser.add_argument('--model', default='/media/NAS/nas_187/siwoo/2023/result/finetune_ori_ins/model', help='')
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
    main(args)