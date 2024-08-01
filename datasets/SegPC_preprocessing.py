import os, argparse
import numpy as np
from PIL import Image
import skimage.io as io
from random import shuffle

def main():
    path= '/media/NAS/nas_70/open_dataset/segpc/TCIA_SegPC_dataset/train'
    test_path = '/media/NAS/nas_70/open_dataset/segpc/TCIA_SegPC_dataset/test'
    save_dir = '/media/NAS/nas_70/open_dataset/segpc/segpc/'
    img_path = os.path.join(path, 'x')
    label_path = os.path.join(path, 'y')

    img_name = os.listdir(img_path)
    shuffle(img_name)
    train_img_name = img_name[:int(len(img_name)*0.8)]
    val_img_name = img_name[int(len(img_name)*0.8):]


    split_patches(img_path, train_img_name, save_dir+'train')
    split_patches(img_path, train_img_name, save_dir+'val')

    split_patches(label_path, val_img_name, save_dir+'train')
    split_patches(label_path, val_img_name, save_dir+'val')

    test_img_path = os.path.join(test_path, 'x')
    test_label_path = os.path.join(test_path, 'y')

    split_patches(test_img_path, val_img_name, save_dir+'test')
    split_patches(test_label_path, val_img_name, save_dir+'test')

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

        return True
    else:
        return False
def split_patches(data_dir, img_name, save_dir, patch_size=250, post_fix="", ext="png"):
    import math
    """ split large image into small patches """
    if create_folder(save_dir):
        print("Spliting large {:s} images into small patches...".format(post_fix))

        image_list = os.listdir(data_dir)
        for image_name in image_list:
            if image_name.startswith("."):
                continue
            name = image_name.split('.')[0]
            if post_fix and name[-len(post_fix):] != post_fix:
                continue
            image_path = os.path.join(data_dir, image_name)
            image = io.imread(image_path)
            seg_imgs = []

            # split into 16 patches of size 250x250
            h, w = image.shape[0], image.shape[1]
            h_num, w_num = np.ceil(h/patch_size), np.ceil(w/patch_size)
            h_overlap = math.ceil((h_num * patch_size - h) / (h_num-1))
            w_overlap = math.ceil((w_num * patch_size - w) / (w_num-1))
            for i in range(0, h - patch_size + 1, patch_size - h_overlap):
                for j in range(0, w - patch_size + 1, patch_size - w_overlap):
                    if len(image.shape) == 3:
                        patch = image[i:i + patch_size, j:j + patch_size, :]
                    else:
                        patch = image[i:i + patch_size, j:j + patch_size]
                    seg_imgs.append(patch)

            for k in range(len(seg_imgs)):
                if post_fix:
                    io.imsave(
                        '{:s}/{:s}_{:d}_{:s}.{:s}'.format(save_dir, name[:-len(post_fix) - 1], k, post_fix, ext),
                        seg_imgs[k])
                else:
                    io.imsave('{:s}/{:s}_{:d}.{:s}'.format(save_dir, name, k, ext), seg_imgs[k])

if __name__ == '__main__':
    import random
    random.seed(777)
    main()