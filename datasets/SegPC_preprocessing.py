import glob
import os, argparse
import numpy as np
from PIL import Image
import skimage.io as io
from random import shuffle

def main(train, test):
    path= '/media/NAS/nas_70/open_dataset/segpc/TCIA_SegPC_dataset/train'
    test_path = '/media/NAS/nas_70/open_dataset/segpc/TCIA_SegPC_dataset/test'
    save_dir = '/media/NAS/nas_70/open_dataset/segpc/segpc/'
    img_path = os.path.join(path, 'x')
    label_path = os.path.join(path, 'y')

    if train == True:
        img_name = os.listdir(img_path)
        shuffle(img_name)
        train_img_name = img_name[:int(len(img_name)*0.8)]
        val_img_name = img_name[int(len(img_name)*0.8):]


        # split_patches(img_path, train_img_name, save_dir+'train/image')
        # split_patches(label_path, val_img_name, save_dir+'val/image')

        split_patches_label(label_path, train_img_name, save_dir+'train/label', version_test=False)
        split_patches_label(label_path, val_img_name, save_dir+'val/label', version_test=False)

    if test == True:
        test_img_path = os.path.join(test_path, 'x')
        test_label_path = os.path.join(test_path, 'y')
        test_img_name = os.listdir(test_img_path)

        # split_patches(test_img_path, val_img_name, save_dir+'test')
        split_patches_label(test_label_path, test_img_name, save_dir+'test/label', version_test=True)

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

        return True
    else:
        return False
def split_patches(data_dir, img_name_list, save_dir, patch_size=1024, post_fix="", ext="png"):
    import math
    """ split large image into small patches """
    if create_folder(save_dir):
        print("Spliting large {:s} images into small patches...".format(post_fix))

        image_list = img_name_list
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

def split_patches_label(data_dir, img_name_list, save_dir, patch_size=1024, post_fix="", ext="png", version_test='False'):
    import math
    """ split large image into small patches """
    if create_folder(save_dir + '_nuclei'):
        pass
    if create_folder(save_dir + '_cell'):
        pass
    print("Spliting large {:s} images into small patches...".format(post_fix))

    for image_name in img_name_list:
        idx_list = glob.glob(os.path.join(data_dir, '*{:s}*'.format(image_name[:-4])))
        index = 1
        for idx in idx_list:
            # image_idx = io.imread(idx, as_gray=True)
            image_idx = Image.open(idx).convert('L')
            image_idx = np.array(image_idx)
            if index == 1:
                n_image = np.zeros_like(image_idx)
                n_point = np.zeros_like(image_idx)
                c_image = np.zeros_like(image_idx)
                c_point = np.zeros_like(image_idx)
            # print(index)
            n_image[image_idx == np.unique(image_idx)[1]] = index
            c_image[image_idx > 0] = index

            coor = np.where(image_idx == np.unique(image_idx)[1])
            y, x = coor
            n_point[round(np.mean(y)), round(np.mean(x))] = 255

            coor = np.where(image_idx > 0)
            y, x = coor
            c_point[round(np.mean(y)), round(np.mean(x))] = 255

            index +=1
        if version_test ==  False:
            n_seg_imgs = []
            c_seg_imgs = []
            n_point_imgs = []
            c_point_imgs = []


            # split into 16 patches of size 250x250
            h, w = n_image.shape[0], n_image.shape[1]
            h_num, w_num = np.ceil(h/patch_size), np.ceil(w/patch_size)
            h_overlap = math.ceil((h_num * patch_size - h) / (h_num-1))
            w_overlap = math.ceil((w_num * patch_size - w) / (w_num-1))
            for i in range(0, h - patch_size + 1, patch_size - h_overlap):
                for j in range(0, w - patch_size + 1, patch_size - w_overlap):
                    if len(n_image.shape) == 3:
                        n_patch = n_image[i:i + patch_size, j:j + patch_size, :]
                        c_patch = c_image[i:i + patch_size, j:j + patch_size, :]
                        n_point = n_image[i:i + patch_size, j:j + patch_size, :]
                        c_point = c_image[i:i + patch_size, j:j + patch_size, :]
                    else:
                        n_patch = n_image[i:i + patch_size, j:j + patch_size].astype(np.uint16)
                        c_patch = c_image[i:i + patch_size, j:j + patch_size].astype(np.uint16)
                        n_point = n_image[i:i + patch_size, j:j + patch_size].astype(np.uint16)
                        c_point = c_image[i:i + patch_size, j:j + patch_size].astype(np.uint16)
                    n_seg_imgs.append(n_patch)
                    c_seg_imgs.append(c_patch)
                    n_point_imgs.append(n_point)
                    c_point_imgs.append(c_point)

            for k in range(len(n_seg_imgs)):
                if post_fix:
                    io.imsave(
                        '{:s}/{:s}_{:d}_{:s}.{:s}'.format(save_dir, image_name[:-len(post_fix) - 1], k, post_fix, ext),
                        n_seg_imgs[k])
                    io.imsave(
                        '{:s}/{:s}_{:d}_{:s}.{:s}'.format(save_dir, image_name[:-len(post_fix) - 1], k, post_fix, ext),
                        c_seg_imgs[k])
                else:
                    io.imsave('{:s}/{:s}_{:d}.{:s}'.format(save_dir+'_nuclei', image_name[:-4], k, ext), n_seg_imgs[k], check_contrast=False)
                    io.imsave('{:s}/{:s}_{:d}.{:s}'.format(save_dir+'_cell', image_name[:-4], k, ext), c_seg_imgs[k], check_contrast=False)
                    io.imsave('{:s}/{:s}_{:d}.{:s}'.format(save_dir+'_point_nuclei', image_name[:-4], k, ext), n_point_imgs[k], check_contrast=False)
                    io.imsave('{:s}/{:s}_{:d}.{:s}'.format(save_dir+'_point_cell', image_name[:-4], k, ext), c_point_imgs[k], check_contrast=False)
        else:
            io.imsave('{:s}/{:s}.{:s}'.format(save_dir + '_nuclei', image_name[:-4], ext), n_image, check_contrast=False)
            io.imsave('{:s}/{:s}.{:s}'.format(save_dir + '_cell', image_name[:-4], ext), c_image, check_contrast=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    import random
    random.seed(777)
    main(train=args.train, test=args.test)
