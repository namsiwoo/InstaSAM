import os, random, json
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image

import torch

from models.segment_anything.build_sam import build_sam
from models.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    mask = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))

    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3)])
        img[m] = color_mask
        mask[m] = i+1

    return img, mask

def make_split(data_root, save_root):
    # img_list = os.listdir(os.path.join(data_root, 'RGB'))
    img_list = os.listdir(os.path.join(data_root, 'images/train'))
    random.shuffle(img_list)

    train_list = img_list[:int(len(img_list)*0.5)]
    val_list = img_list[int(len(img_list)*0.5): int(len(img_list)*0.7)]
    test_list = img_list[int(len(img_list)*0.7):]

    json_data = OrderedDict()
    json_data['train'] = train_list
    json_data['val'] = val_list
    json_data['test'] = test_list

    with open('{:s}/train_val_test.json'.format(save_root), 'w') as make_file:
        json.dump(json_data, make_file, indent='\t')
def main(data_root, save_root):
    # img_folder = 'RGB'
    img_folder = 'images/train'

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    sam_checkpoint = '/media/NAS/nas_187/siwoo/2023/SAM model/SAM-Adapter-PyTorch-main/sam_vit_h_4b8939.pth'

    sam = build_sam(sam_checkpoint)


    # mask_generator = SAM2AutomaticMaskGenerator(sam2)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    img_list = os.listdir(os.path.join(data_root, img_folder))
    for img_name in img_list:
        print(img_name, '...')
        img_path = os.path.join(data_root, img_folder, img_name)
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = mask_generator.generate(img)
        img, mask = show_anns(mask)

        img = Image.fromarray((img*255).astype(np.uint8))
        img.save(os.path.join(save_root, 'vis', img_name))

        mask = Image.fromarray((mask).astype(np.uint16))
        mask.save(os.path.join(save_root, 'masks', img_name))


if __name__ == '__main__':
    # data_root = '/media/NAS/nas_187/datasets/synthia/RAND_CITYSCAPES'
    # save_root = '/media/NAS/nas_70/siwoo_data/UDA_citycapes/synthia'
    data_root = '/media/NAS/nas_70/open_dataset/TNBC/TNBC/via instance learning data_for_train/TNBC'
    save_root = '/media/NAS/nas_70/siwoo_data/UDA_citycapes/TNBC_sam'
    # make_split(data_root, save_root)
    main(data_root, save_root)

    #CUDA_VISIBLE_DEVICES=0 python make_segment.py