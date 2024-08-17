import os
import glob
import shutil
import numpy as np
from PIL import Image
import tifffile as tiff
from skimage import measure

MoNuSAC_path_root = '/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC_ori/MoNuSAC_images_and_annotations'
MoNuSAC_patch_save = '/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC/images/train'
MoNuSAC_mask_root = '/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC_ori/MoNuSAC_train_mask'
MoNuSAC_mask_save = '/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC/labels_instance/train'
MoNuSAC_point_save = '/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC/labels_point/train'


# MoNuSAC_path_root = '/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC_ori/MoNuSAC Testing Data and Annotations'
# MoNuSAC_patch_save = '/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC/test/images'
# MoNuSAC_mask_root = '/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC_ori/MoNuSAC_test_mask'
# MoNuSAC_mask_save = '/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC/test/labels_instance'
# MoNuSAC_point_save = '/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC/test/labels_point'

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

img_list = glob.glob(os.path.join(MoNuSAC_path_root, '*/*.tif'))
for img in img_list:
    file_name = img.split('/')[-2]+'/'+img.split('/')[-1][:-4]

    class_name = os.listdir(os.path.join(MoNuSAC_mask_root, file_name))
    mask = []
    img_name = img.split('/')[-1][:-4]
    for c in class_name:
        # print(os.listdir(os.path.join(MoNuSAC_mask_root, file_name, c)))
        try:
            # print(os.listdir(os.path.join(MoNuSAC_mask_root, file_name, c)))
            f_img_name = os.listdir(os.path.join(MoNuSAC_mask_root, file_name, c))[0]
        except:
            f_img_name = None
        if f_img_name is not None:
            img_path = os.path.join(MoNuSAC_mask_root, file_name, c, f_img_name)
            mask_1 = tiff.imread(img_path)
            mask.append(np.array(mask_1))

    mask = np.sum(np.array(mask), axis=0)
    mask = measure.label(mask)

    point = np.zeros_like(mask)
    for index in np.unique(mask)[1:]:
        coor = np.where(mask == index)
        y, x = coor
        point[round(np.mean(y)), round(np.mean(x))] = 255

    image = tiff.imread(img)
    image = np.array(image)
    image = Image.fromarray(image.astype(np.uint8))

    mask = Image.fromarray(point.astype(np.uint16))

    point = Image.fromarray(point.astype(np.uint8))
    print(np.array(image).shape)

    if np.array(image).shape[0] < 224:
        t = round((224-np.array(image).shape[0]) / 2)
        b = 224 - (t+np.array(image).shape[0])
        if np.array(image).shape[1] < 224:
            r = round((224-np.array(image).shape[1])/2)
            l = 224-(r+np.array(image).shape[1])
            image = add_margin(image, top=t, right=r, bottom=b, left=l, color=(0,0,0))
        else:
            r, l = 0, 0
            image = add_margin(image, top=t, right=r, bottom=b, left=l, color=(0,0,0))

    else:
        if np.array(image).shape[1] < 224:
            t, b, = 0, 0
            r = round(np.array(image).shape[0]/2)
            l = 224-r
            image = add_margin(image, top=t, right=r, bottom=b, left=l, color=(0,0,0))

    image.save(os.path.join(MoNuSAC_patch_save, img_name)+'.png')
    # shutil.copy(img, os.path.join(MoNuSAC_patch_save, img_name)[:-3]+'.png')
    mask.save(os.path.join(MoNuSAC_mask_save, img_name+'.png'))

    point.save(os.path.join(MoNuSAC_point_save, img_name+'.png'))



    # np.save(os.path.join(MoNuSAC_mask_save, img_name), mask)
