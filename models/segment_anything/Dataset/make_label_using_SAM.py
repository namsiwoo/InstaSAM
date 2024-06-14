import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import skimage.morphology as ski_morph
from torch.nn import functional as F
from scipy.ndimage.morphology import binary_fill_holes
from skimage import measure
from segment_anything.utils.transforms import ResizeLongestSide

def add_margin(pil_img, top, right, bottom, left, color):
    """pading on PIL image"""
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result
def preprocess(x, device):
    """Normalize pixel values and pad to a square input."""
    # x : 1, 3, H, W

    image_encoder_img_size =1024

    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).to(device)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).to(device)
    pixel_mean = pixel_mean.view(-1, 1, 1)
    pixel_std = pixel_std.view(-1, 1, 1)


    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = image_encoder_img_size - h
    padw = image_encoder_img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def norm(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def torch_norm(img):
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))

if __name__ == '__main__':
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    image_root = '/media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg/images/train'
    instance_root = '/media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg/labels_instance/train'
    point_root = '/media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg/labels_point/train'
    voronoi_root = '/media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg/labels_voronoi/train'
    mask_root = '/media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg/labels_instance/train'

    # save_root = '/media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg/labels_sam/train'
    save_root = '/media/NAS/nas_187/siwoo/2023/result/sem-ins_co_train'

    # image_root = '/media/NAS/nas_187/PATHOLOGY_DATA/CoNSeP/CoNSeP/via instance learning data_for_train/CoNSeP/images/train'
    # instance_root = '/media/NAS/nas_187/PATHOLOGY_DATA/CoNSeP/CoNSeP/via instance learning data_for_train/CoNSeP/labels_instance/train'
    # point_root = '/media/NAS/nas_187/PATHOLOGY_DATA/CoNSeP/CoNSeP/via instance learning data_for_train/CoNSeP/labels_point/train'
    # voronoi_root = '/media/NAS/nas_187/PATHOLOGY_DATA/CoNSeP/CoNSeP/via instance learning data_for_train/CoNSeP/labels_voronoi/train'
    # mask_root = '/media/NAS/nas_187/PATHOLOGY_DATA/CoNSeP/CoNSeP/via instance learning data_for_train/CoNSeP/labels_instance/train'
    #
    # save_root = '/media/NAS/nas_187/PATHOLOGY_DATA/CoNSeP/CoNSeP/via instance learning data_for_train/CoNSeP/labels_sam/train'

    from segment_anything import sam_model_registry

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_model = sam_model.to(device)
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=0.0001)
    # loss_fn = torch.nn.MSELoss(reduction='none')
    # loss_fn = torch.nn.NLLLoss(ignore_index=2).to(device)

    img_list = os.listdir(image_root)
    img_list.sort()
    point_list = os.listdir(point_root)
    point_list.sort()

    transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    # for epoch in range(100):
    #     os.makedirs('/media/NAS/nas_187/siwoo/2023/result/sem-ins_co_train/train/img/point/'+str(epoch), exist_ok=True)
    #     os.makedirs('/media/NAS/nas_187/siwoo/2023/result/sem-ins_co_train/train/img/box/'+str(epoch), exist_ok=True)

        # losses = 0
        # loss_map = np.zeros((256, 256))

    for j in range(len(img_list)):
        losses = 0
        losses_map = np.zeros((256, 256))

        # image
        # img_list[j]= 'Breast_TCGA-AR-A1AK-01Z-00-DX1_1.png'
        image = Image.open(os.path.join(image_root, img_list[j])).convert('RGB')
        # image = add_margin(image, 3, 3, 3, 3, (0, 0, 0))
        image = np.array(image)

        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image_torch = preprocess(input_image_torch, device)

        print(np.unique(input_image_torch[0].detach().cpu().numpy().transpose(1, 2, 0)))
        plt.imshow(input_image_torch[0].detach().cpu().numpy().transpose(1, 2, 0))
        plt.savefig('/media/NAS/nas_187/siwoo/2023/result/finetune/img/0/input_torch_ex.png')

    #    print(aaaa)

        # image embedding
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image_torch)


        mask = Image.open(os.path.join(instance_root, img_list[j][:-4] + '_label.png'))
        # mask = add_margin(mask, 3, 3, 3, 3, 0)
        mask = np.array(mask)
        mask = ski_morph.label(mask)

        bnd_box = np.zeros((len(np.unique(mask)[1:]), 4))
        # print('------')
        # print(np.unique(mask), len(np.unique(mask)[1:]))
        bbbb_box = np.zeros((250, 250))
        for iter, k in enumerate(np.unique(mask)[1:]):
            coor = np.where(mask==k)
            x_coor = coor[1]
            y_coor = coor[0]

            x0 = np.min(x_coor)
            y0 = np.min(y_coor)
            x1 = np.max(x_coor)
            y1 = np.max(y_coor)

            w = np.max(x_coor) - np.min(x_coor)
            h = np.max(y_coor) - np.min(y_coor)
            bnd_box[iter] = np.array((x0, y0, x1, y1))
            bbbb_box[y0:h + y0, x0:w + x0] = iter + 1

        bnd_box = transform.apply_boxes(bnd_box, (250, 250))
        box_torch = torch.as_tensor(bnd_box, dtype=torch.float, device=device)

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )

            low_res_masks, _ = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
        upscaled_masks_b = sam_model.postprocess_masks(low_res_masks, (1024, 1024), (250, 250)).to(device)
        from torch.nn.functional import threshold, normalize
        binary_mask = normalize(threshold(upscaled_masks_b, 0.0, 0)).to(device)

        gt = np.zeros((250, 250))
        for a in range(binary_mask.shape[0]):
            gt[binary_mask[a, 0, :, :].detach().cpu().numpy()!=0] = 255

        gt = Image.fromarray(gt.astype(np.uint8))
        gt.save('/media/NAS/nas_187/siwoo/2023/SAM_pseudo_label/Box_annotation/'+img_list[j])

        #
        #     with torch.no_grad():
        #         sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        #             points=None,
        #             boxes=None,
        #             masks=None,
        #         )
        #
        #     low_res_masks, _ = sam_model.mask_decoder(
        #         image_embeddings=image_embedding,
        #         image_pe=sam_model.prompt_encoder.get_dense_pe(),
        #         sparse_prompt_embeddings=sparse_embeddings,
        #         dense_prompt_embeddings=dense_embeddings,
        #         multimask_output=False,
        #     )
        #
        #     upscaled_masks = sam_model.postprocess_masks(low_res_masks, (1024, 1024), (256, 256)).to(device)
        #
        #
        #
        #
        #     from torch.nn.functional import threshold, normalize
        #     binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(device)
        #
        #     # for k in range(upscaled_masks.detach().cpu().shape[0]):
        #     #     plt.subplot(3, int(upscaled_masks.detach().cpu().shape[0]/3)+1, k+1)
        #     #     plt.imshow(binary_mask[k][0].detach().cpu().numpy())
        #     # plt.savefig('/media/NAS/nas_187/siwoo/2023/result/sem-ins_co_train/prompt_point/' + 'binary_' + img_list[j])
        #
        #     for k in range(upscaled_masks.detach().cpu().shape[0]):
        #         if iou_predictions[k][0] < 0.90 or np.sum(binary_mask[k][0].detach().cpu().numpy() != 0) > 256 * 256 / 6:
        #             upscaled_masks[k, 0, :, :] = -1
        #             # print(np.unique(upscaled_masks.detach().cpu()[k, 0]))
        #     filtered_mask = torch.argmax(torch.cat([upscaled_masks, torch.zeros(1, 1, 256, 256).to(device)], 0), dim=0)[0]
        #     filtered_mask = filtered_mask.detach().cpu().numpy()
        #
        #     min_area = 20
        #     fg_mask = ski_morph.remove_small_objects(filtered_mask, min_area)
        #     fg_mask = binary_fill_holes(fg_mask > 0)
        #     filtered_mask[fg_mask==0]=0
        #
        #
        #     # for k in range(upscaled_masks.detach().cpu().shape[0]):
        #     #     plt.subplot(3, int(upscaled_masks.detach().cpu().shape[0]/3)+1, k+1)
        #     #     plt.imshow(upscaled_masks[k][0].detach().cpu().numpy())
        #     #     plt.colorbar()
        #     #     plt.title('iou: '+ str(round(iou_predictions.detach().cpu().numpy()[k][0], 3)), fontsize=7)
        #     # # plt.savefig('/media/NAS/nas_187/siwoo/2023/result/sem-ins_co_train/prompt_point/'+img_list[j])
        #     # plt.savefig('/media/NAS/nas_187/siwoo/2023/result/sem-ins_co_train/train/img/point/'+str(epoch) + '/pred_upscale_'+str(j) + img_list[j])
        #
        #
        #     #
        #     # plt.subplot(1, 3, 1)
        #     # plt.imshow(image)
        #     # plt.subplot(1, 3, 2)
        #     # plt.imshow(point_image)
        #     # plt.subplot(1, 3, 3)
        #     # plt.imshow(filtered_mask)
        #     # plt.savefig('/media/NAS/nas_187/siwoo/2023/result/sem-ins_co_train/prompt_point/' + 'pred_' + img_list[j])
        #
        # ##############################################################################################################################################################
        # ##############################################################################################################################################################
        # ##############################################################################################################################################################
        #     # to make bnd box
        #     # mask = Image.open(os.path.join(instance_root, img_list[j][:-4] + '_label.png'))
        #     # mask = np.array(mask)
        #     # mask = ski_morph.label(mask)
        #
        #     # bnd_box = np.zeros((len(np.unique(filtered_mask)), 4))
        #
        #     if len(np.unique(filtered_mask)) > 1:
        #         bbbb_box = np.zeros((binary_mask.shape[2], binary_mask.shape[3]))
        #
        #         ignored = upscaled_masks.shape[0]
        #         loss_fn = torch.nn.NLLLoss(ignore_index=ignored, reduction='none').to(device)
        #         pseudo_gt = torch.zeros(1, 1, 256, 256).to(device) + ignored
        #         for k in (np.unique(filtered_mask)[:-1]):
        #             bnd_box = np.zeros((20, 4))
        #             coor = np.where(filtered_mask == k)
        #             # coor = np.where(mask==k+1)
        #             x_coor = coor[1]
        #             y_coor = coor[0]
        #             for loop in range(20):
        #
        #                 x0 = max(0, np.min(x_coor) + random.randrange(-3, 4))
        #                 y0 = max(0, np.min(y_coor) + random.randrange(-3, 4))
        #                 x1 = min(np.max(x_coor) + random.randrange(-3, 4), 256)
        #                 y1 = min(np.max(y_coor) + random.randrange(-3, 4), 256)
        #
        #                 w = np.max(x_coor)-np.min(x_coor)
        #                 h = np.max(y_coor)-np.min(y_coor)
        #
        #
        #                 bnd_box[loop] = np.array((x0, y0, x1, y1))
        #
        #             bnd_box = transform.apply_boxes(bnd_box, (256, 256))
        #             box_torch = torch.as_tensor(bnd_box, dtype=torch.float, device=device)
        #
        #             with torch.no_grad():
        #                 sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        #                     points=None,
        #                     boxes=box_torch,
        #                     masks=None,
        #                 )
        #
        #                 low_res_masks, iou_predictions = sam_model.mask_decoder(
        #                     image_embeddings=image_embedding,
        #                     image_pe=sam_model.prompt_encoder.get_dense_pe(),
        #                     sparse_prompt_embeddings=sparse_embeddings,
        #                     dense_prompt_embeddings=dense_embeddings,
        #                     multimask_output=False,
        #                 )
        #
        #                 upscaled_masks_m = sam_model.postprocess_masks(low_res_masks, (1024, 1024), (256, 256)).to(device)
        #             # print(upscaled_masks.shape)
        #
        #             # for k in range(upscaled_masks.detach().cpu().shape[0]):
        #             #     plt.subplot(3, int(upscaled_masks.detach().cpu().shape[0]/3)+1, k+1)
        #             #     plt.imshow(upscaled_masks[k][0].detach().cpu().numpy())
        #             #     plt.title('iou: '+ str(round(iou_predictions.detach().cpu().numpy()[k][0], 3)), fontsize=7)
        #             # plt.savefig('/media/NAS/nas_187/siwoo/2023/result/sem-ins_co_train/prompt_box/'+img_list[j])
        #             #
        #             # binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(device)
        #             #
        #             # for k in range(upscaled_masks.detach().cpu().shape[0]):
        #             #     plt.subplot(3, int(upscaled_masks.detach().cpu().shape[0]/3)+1, k+1)
        #             #     plt.imshow(binary_mask[k][0].detach().cpu().numpy())
        #             # plt.savefig('/media/NAS/nas_187/siwoo/2023/result/sem-ins_co_train/prompt_box/' + 'binary_' + img_list[j])
        #             #
        #             # filtered_mask = torch.argmax(torch.cat([torch.zeros(1, 1, 256, 256).to(device), upscaled_masks], 0), dim=0)[0]
        #             # filtered_mask = filtered_mask.detach().cpu().numpy()
        #             #
        #             # min_area = 20
        #             # fg_mask = ski_morph.remove_small_objects(filtered_mask, min_area)
        #             # fg_mask = binary_fill_holes(fg_mask > 0)
        #             # filtered_mask[fg_mask==0]=0
        #
        #             # plt.subplot(1, 3, 1)
        #             # plt.imshow(image)
        #             # plt.subplot(1, 3, 2)
        #             # plt.imshow(bbbb_box)
        #             # plt.subplot(1, 3, 3)
        #             # plt.imshow(filtered_mask)
        #             # plt.savefig('/media/NAS/nas_187/siwoo/2023/result/sem-ins_co_train/prompt_box/' + 'pred_' + img_list[j])
        #
        #             mean_pred = torch.mean(upscaled_masks_m, dim=0)[0]
        #             mean_var = torch.var(upscaled_masks_m, dim=0)[0]
        #
        #             pred = norm(mean_pred.detach().cpu().numpy())>0.5
        #             ignored_region = norm(mean_var.detach().cpu().numpy())>0.3
        #
        #             pseudo_gt[0, 0, pred==1] = k
        #             pseudo_gt[0, 0, ignored_region] = ignored
        #
        #
        #         bnd_box = np.zeros((len(np.unique(filtered_mask[:-1])), 4))
        #
        #         print(box_torch.shape, upscaled_masks_m.shape, '--------')
        #
        #         log_prob_maps = F.log_softmax(upscaled_masks.squeeze(1), dim=0)
        #         log_prob_maps = torch.unsqueeze(log_prob_maps, 0)
        #         #
        #         # gt = torch.from_numpy(pred).float().to(device)
        #         # gt[ignored!=0] = 2
        #         # print(np.unique(gt.detach().cpu().numpy()))
        #         # gt = torch.unsqueeze(gt, 0)
        #         #
        #         # loss = loss_fn(log_prob_maps, gt.long())
        #         # loss_map[:, :, ignored!=0]=0
        #         # loss = loss_map.mean
        #
        #
        #         print(np.unique(filtered_mask), upscaled_masks.shape, np.unique(pseudo_gt.detach().cpu().numpy()))
        #         lossmap = loss_fn(log_prob_maps, pseudo_gt.squeeze(0).long().to(device))
        #         loss =lossmap.mean()
        #
        #
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         losses += loss.item()
        #
        #         filtered_mask2 = torch.argmax(torch.cat([torch.zeros(1, 1, 256, 256).to(device), upscaled_masks_m], 0), dim=0)[0]
        #         filtered_mask2 = filtered_mask2.detach().cpu().numpy()
        #
        #         min_area = 20
        #         fg_mask = ski_morph.remove_small_objects(filtered_mask2, min_area)
        #         fg_mask = binary_fill_holes(fg_mask > 0)
        #         filtered_mask2[fg_mask==0]=0
        #
        #         # plt.subplot(1, 3, 1)
        #         # plt.imshow(filtered_mask)
        #         # plt.subplot(1, 3, 2)
        #         # plt.imshow(pseudo_gt.detach().cpu()[0][0])
        #         # plt.colorbar()
        #         # plt.subplot(1, 3, 3)
        #         # plt.imshow(lossmap[0].detach().cpu())
        #         # plt.colorbar()
        #         # plt.savefig('/media/NAS/nas_187/siwoo/2023/result/sem-ins_co_train/train/img/ex.png')
        #
        #
        #
        #
        #                 # losses_map += loss_map.detach().cpu().numpy()
        #                 # losses += loss
        #
        #                 # plt.subplot(2, 3, 1)
        #                 # plt.imshow(image)
        #                 # plt.subplot(2, 3, 2)
        #                 # plt.imshow(mean_pred.detach().cpu().numpy())
        #                 # plt.colorbar()
        #                 # plt.subplot(2, 3, 5)
        #                 # plt.imshow(norm(mean_pred.detach().cpu().numpy())>0.5)
        #                 # plt.subplot(2, 3, 3)
        #                 # plt.imshow(mean_var.detach().cpu().numpy())
        #                 # plt.colorbar()
        #                 # plt.subplot(2, 3, 6)
        #                 # plt.imshow(norm(mean_var.detach().cpu().numpy())>0.3)
        #                 # plt.savefig('/media/NAS/nas_187/siwoo/2023/result/sem-ins_co_train/prompt_box/' + str(iter)+'_pred_' + img_list[j])
        #
        #         print(loss.item())
        #
        #         plt.subplot(1, 5, 1)
        #         plt.imshow(image)
        #         plt.axis('off')
        #         plt.subplot(1, 5, 2)
        #         plt.imshow(point_image)
        #         plt.axis('off')
        #         plt.subplot(1, 5, 3)
        #         plt.imshow(filtered_mask)
        #         plt.axis('off')
        #         plt.subplot(1, 5, 4)
        #         plt.imshow(pseudo_gt.detach().cpu()[0][0])
        #         plt.axis('off')
        #         plt.subplot(1, 5, 5)
        #         plt.imshow(lossmap[0].detach().cpu())
        #         plt.axis('off')
        #         plt.savefig('/media/NAS/nas_187/siwoo/2023/result/sem-ins_co_train/train/img/point/'+str(epoch) + '/pred_'+str(j) + img_list[j])
        #
        #         plt.subplot(1, 3, 1)
        #         plt.imshow(image)
        #         plt.axis('off')
        #         plt.subplot(1, 3, 2)
        #         plt.imshow(bbbb_box)
        #         plt.axis('off')
        #         plt.subplot(1, 3, 3)
        #         plt.imshow(filtered_mask2)
        #         plt.axis('off')
        #         # plt.subplot(1, 4, 4)
        #         # plt.imshow(losses_map)
        #         plt.axis('off')
        #         plt.savefig('/media/NAS/nas_187/siwoo/2023/result/sem-ins_co_train/train/img/box/'+str(epoch) + '/pred_'+str(j) + img_list[j])
        #
        #         del pseudo_gt, mean_pred, mean_var, pred, ignored, filtered_mask, filtered_mask2
        #         torch.cuda.empty_cache()
        #
        # torch.save(sam_model.state_dict(), '/media/NAS/nas_187/siwoo/2023/result/sem-ins_co_train/train/model/'+str(epoch)+'_model.pth')
        #
        # # loss = loss_fn(binary_mask, gt_binary_mask)
        # # optimizer.zero_grad()
        # # loss.backward()
        # # optimizer.step()
