import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data

from PIL import Image
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from scipy.ndimage.morphology import binary_dilation
import skimage.morphology, skimage.measure


class Crop_dataset(torch.utils.data.Dataset): #MO, CPM, CoNSeP
    def __init__(self, args, split, use_mask=False, patch=False):
        self.args = args
        self.root_dir = os.path.expanduser(self.args.data_path)  # /media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg
        self.split = split
        self.use_mask = use_mask

        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

        # create image augmentation
        from datasets.get_transforms_ssl import get_transforms

        if self.split == 'train':
            self.transform = get_transforms({
                # 'random_resize': [0.8, 1.25],
                'horizontal_flip': True,
                'random_affine': 0.3,
                'random_rotation': 90,
                'random_crop': 224,
                'label_encoding': [0, 1], #new_label: 3 else 2
                'to_tensor': 1, # number of img
                'normalize': np.array([self.mean, self.std])
            })
        else:
            self.transform = get_transforms({
                'to_tensor': 1,
                'normalize': np.array([self.mean, self.std])
            })

        # read samples
        self.samples = self.read_samples(self.root_dir, self.split, few_shot=args.few_shot)

        # set num samples
        self.num_samples = len(self.samples)
        print('{} dataset {} loaded'.format(self.split, self.num_samples))

    def read_samples(self, root_dir, split, few_shot=False):
        if split == 'train':
            if few_shot==False:
                samples = os.listdir(os.path.join(root_dir, 'images', split))
            else:
                samples = os.listdir(os.path.join(root_dir, 'images', 'train_few_shot'))
            # samples = os.listdir(os.path.join('/media/NAS/nas_32/siwoo/CPM/train'))

        else:
            root_dir = self.root_dir.split('/')
            new_dir = ''
            for dir in root_dir[:-2]:
                new_dir += dir + '/'
            with open(os.path.join(new_dir, 'train_val_test.json')) as f:
                split_dict = json.load(f)
            filename_list = split_dict[split]
            samples = [os.path.join(f) for f in filename_list]

        # samples = os.listdir(os.path.join(root_dir, 'images', split))
        return samples

    def __getitem__(self, index):
        img_name = self.samples[index % len(self.samples)]

        if self.split == 'train':
            # 1) read image
            img = Image.open(os.path.join(self.root_dir, 'images', self.split, img_name)).convert('RGB')

            point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name)).convert('L')
            # point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name[:-4] + '_label_point.png')).convert('L')
            point = binary_dilation(np.array(point), iterations=2)
            point = Image.fromarray(point)

            if self.use_mask == True:
                box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name)))
                # box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4]+'_label.png')))
                box_label = skimage.morphology.label(box_label)
                box_label = Image.fromarray(box_label.astype(np.uint16))

                sample = [img, box_label, point]#, cluster_label, voronoi_label]  # , new_mask
            else:
                sample = [img, point]
            sample = self.transform(sample)

        else:
            root_dir = self.root_dir.split('/')
            new_dir = ''
            for dir in root_dir[:-2]:
                new_dir += dir + '/'

            img = Image.open(os.path.join(new_dir, 'images', img_name)).convert('RGB')
            mask = Image.open(os.path.join(new_dir, 'labels_instance', img_name[:-4] + '_label.png'))

            sample = [img, mask]
            sample = self.transform(sample)

        return sample, str(img_name[:-4])

    def __len__(self):
        return self.num_samples

class MoNuSeg_weak_dataset(torch.utils.data.Dataset):
    def __init__(self, args, split, ssl=False):
        self.args = args
        self.root_dir = os.path.expanduser(self.args.data)  # /media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg
        self.split = split
        self.ssl = ssl
        if self.ssl == False:
            from datasets.get_transforms import get_transforms
        else:
            from datasets.get_transforms_ssl import get_transforms

        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

        # Imagenet norm?
        # self.mean = np.array([0.485,0.456,0.406])
        # self.std = np.array([0.229,0.224,0.225])

        if 'CPM' in self.root_dir:
            r = 6
        else:
            r = 8

        # create image augmentation
        if self.split == 'train':
            self.transform = get_transforms({
                # 'random_resize': [0.8, 1.25],
                'horizontal_flip': True,
                'random_affine': 0.3,
                'random_rotation': 90,
                'random_crop': 224,
                'label_encoding': [0, 1, r], #new_label: 3 else 2
                'to_tensor': 1, # number of img
                'normalize': np.array([self.mean, self.std])
            })
        else:
            self.transform = get_transforms({
                'to_tensor': 1,
                'normalize': np.array([self.mean, self.std])
            })

        # read samples
        self.samples = self.read_samples(self.root_dir, self.split)

        # self.samples = self.read_samples('/media/NAS/nas_70/open_dataset/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg', self.split)
        # self.samples += self.read_samples('/media/NAS/nas_70/open_dataset/CPM/CPM 17/via instance learning data_for_train/CPM 17', self.split)
        # self.samples += self.read_samples('/media/NAS/nas_32/siwoo/TNBC/TNBC/via instance learning data_for_train/TNBC', self.split)

        # set num samples
        self.num_samples = len(self.samples)

        # print('{} dataset {} loaded'.format(self.split, self.num_samples))

    def read_samples(self, root_dir, split):
        # if split == 'train':
        #     samples = os.listdir(os.path.join(root_dir, 'images', split))
        #     # samples = os.listdir(os.path.join('/media/NAS/nas_70/open_dataset/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg', 'images', 'train_few_shot'))
        #     # samples = os.listdir(os.path.join('/media/NAS/nas_32/siwoo/CPM/train'))
        #
        # else:
        #     root_dir = self.root_dir.split('/')
        #     new_dir = ''
        #     for dir in root_dir[:-2]:
        #         new_dir += dir + '/'
        #     with open(os.path.join(new_dir, 'train_val_test.json')) as f:
        #         split_dict = json.load(f)
        #     filename_list = split_dict[split]
        #     samples = [os.path.join(f) for f in filename_list]

        samples = os.listdir(os.path.join(root_dir, 'images', split))
        return samples

    def __getitem__(self, index):
        img_name = self.samples[index % len(self.samples)]

        if self.split == 'train':
            # 1) read image
            img = Image.open(os.path.join(self.root_dir, 'images', self.split, img_name)).convert('RGB')

            if self.ssl == False:

                # box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4]+'_label.png')))
                box_label = np.array(Image.open(os.path.join('/media/NAS/nas_187/siwoo/train', 'labels_instance', self.split, img_name[:-4]+'_label.png')))
                box_label = skimage.morphology.label(box_label)
                box_label = Image.fromarray(box_label.astype(np.uint16))

                # 3) do image augmentation
                sample = [img, box_label]  # , new_mask
                sample = self.transform(sample)
            else:
                box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name)))
                # box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4]+'_label.png')))
                box_label = skimage.morphology.label(box_label)
                box_label = Image.fromarray(box_label.astype(np.uint16))

                # box_label = Image.open(os.path.join('/media/NAS/nas_187/siwoo/2023/SAM_pseudo_label/Box_annotation', img_name))
                # box_label = Image.open(os.path.join('/media/NAS/nas_187/siwoo/2023/SAM_pseudo_label/Box_annotation_CPM', img_name))

                # try:
                #     box_label = Image.open(os.path.join(self.root_dir, 'labels_cluster', self.split, img_name[:-4]+'_label_cluster.png'))
                # except:
                #     try:
                #         box_label = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/CPM/CPM 17/via instance learning data_for_train/CPM 17', 'labels_cluster', self.split, img_name[:-4] + '_label_cluster.png'))
                #     except:
                #         box_label = Image.open(os.path.join('/media/NAS/nas_32/siwoo/TNBC/TNBC/via instance learning data_for_train/TNBC', 'labels_cluster', self.split, img_name[:-4] + '_label_cluster.png'))
                #
                #
                # try:
                #     point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name[:-4] + '_label_point.png')).convert('L')
                # except:
                #     try:
                #         point = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/CPM/CPM 17/via instance learning data_for_train/CPM 17', 'labels_point', self.split, img_name[:-4] + '_label_point.png')).convert('L')
                #     except:
                #         point = Image.open(os.path.join('/media/NAS/nas_32/siwoo/TNBC/TNBC/via instance learning data_for_train/TNBC', 'labels_point', self.split, img_name[:-4] + '_label_point.png')).convert('L')

                point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name)).convert('L')
                # point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name[:-4] + '_label_point.png')).convert('L')
                point = binary_dilation(np.array(point), iterations=2)
                point = Image.fromarray(point)

                # cluster_label = Image.open(os.path.join(self.root_dir, 'labels_cluster', self.split, img_name[:-4]+'_label_cluster.png')).convert('RGB')
                # voronoi_label = Image.open(os.path.join(self.root_dir, 'labels_voronoi', self.split, img_name[:-4] + '_label_vor.png')).convert('RGB')
                # cluster_label = Image.open(os.path.join(self.root_dir, 'labels_geo_cluster', self.split, img_name[:-4]+'_label_geo_cluster.png')).convert('RGB')
                # voronoi_label = Image.open(os.path.join(self.root_dir, 'labels_geo_voronoi', self.split, img_name[:-4] + '_label_geo_vor.png')).convert('RGB')

                sample = [img, box_label, point]#, cluster_label, voronoi_label]  # , new_mask
                sample = self.transform(sample)


        else:
            # mask = Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4]+'_label.png')).convert('L')
            # mask = np.array(mask)
            root_dir = self.root_dir.split('/')
            new_dir = ''
            for dir in root_dir[:-2]:
                new_dir += dir + '/'

            # img = Image.open(os.path.join(new_dir, 'images', img_name)).convert('RGB')
            # mask = Image.open(os.path.join(new_dir, 'labels_instance', img_name[:-4] + '_label.png'))

            # img = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/images/val', img_name)).convert('RGB')
            # mask = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/labels_instance/val', img_name))
            img = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC/images/val', img_name)).convert('RGB')
            mask = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC/labels_instance/val2', img_name))


            sample = [img, mask]
            sample = self.transform(sample)

        return sample, str(img_name[:-4])

    def __len__(self):
        return self.num_samples


class MoNuSeg_dataset_coarse_label(torch.utils.data.Dataset):
    def __init__(self, args, split):
        from get_transforms_ori import get_transforms
        self.args = args
        self.root_dir = os.path.expanduser(self.args.data)  # /media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg
        self.split = split

        mean_std = np.load(os.path.join(self.root_dir, 'mean_std.npy'))
        self.mean = np.array(mean_std[0], np.float32)
        self.std = np.array(mean_std[1], np.float32)

        if 'CPM' in self.root_dir:
            r = 6
        else:
            r = 8

        # create image augmentation
        if self.split == 'train':
            self.transform = get_transforms({
                'random_resize': [0.8, 1.25],
                'horizontal_flip': True,
                'random_affine': 0.3,
                'random_rotation': 90,
                'random_crop': 224,
                'label_encoding': [1, r, 3], #new_label: 3 else 2
                'to_tensor': 3, #new_label: 3 else 2
                'normalize': np.array([self.mean, self.std])
            })
        else:
            self.transform = get_transforms({
                'to_tensor': 1,
                'normalize': np.array([self.mean, self.std])
            })

        # read samples
        self.samples = self.read_samples(self.root_dir, self.split)

        # set num samples
        self.num_samples = len(self.samples)

        print('{} dataset {} loaded'.format(self.split, self.num_samples))

    def read_samples(self, root_dir, split):
        if split == 'train':
            samples = os.listdir(os.path.join(root_dir, 'images', split))
        else:
            root_dir = self.root_dir.split('/')
            new_dir = ''
            for dir in root_dir[:-2]:
                new_dir += dir + '/'
            with open(os.path.join(new_dir, 'train_val_test.json')) as f:
                split_dict = json.load(f)
            filename_list = split_dict[split]
            samples = [os.path.join(f) for f in filename_list]
        return samples

    def __getitem__(self, index):
        img_name = self.samples[index % len(self.samples)]

        if self.split == 'train':
            # 1) read image
            # img = Image.open(os.path.join(self.root_dir, 'images', self.split, img_name)).convert('RGB')
            img = Image.open(os.path.join(self.root_dir, 'images', self.split, img_name)).convert('RGB')

            # 2) read point
            point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name[:-4] + '_label_point.png')).convert('L')
            point = binary_dilation(np.array(point), iterations=2)
            point = Image.fromarray(point)

            # 3) read segm
            # cluster_label = Image.open(os.path.join(self.root_dir, 'labels_cluster', self.split, img_name[:-4]+'_label_cluster.png')).convert('RGB')
            # cluster_label = Image.open(os.path.join(self.root_dir, 'labels_geo_cluster', self.split,img_name[:-4] + '_label_geo_cluster.png')).convert('RGB')
            cluster_label = Image.open(os.path.join(self.root_dir, 'labels_sam', self.split,img_name[:-4] + '_label_sam.png')).convert('RGB')


            voronoi_label = Image.open(os.path.join(self.root_dir, 'labels_voronoi', self.split,img_name[:-4]+'_label_vor.png')).convert('RGB')
            # voronoi_label = Image.open(os.path.join(self.root_dir, 'labels_geo_voronoi', self.split, img_name[:-4] + '_label_geo_vor.png')).convert('RGB')

            mask = Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4] + '_label.png'))
            mask = np.array(mask)
            new_mask = np.zeros_like(mask)
            for i in np.unique(mask)[1:]:
                new_mask += skimage.morphology.binary_erosion(mask==i)
            new_mask = Image.fromarray(new_mask.astype(np.uint8))

            # 3) do image augmentation
            sample = [img, point, voronoi_label, cluster_label, new_mask]  # , new_mask
            sample = self.transform(sample)


        else:
            # mask = Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4]+'_label.png')).convert('L')
            # mask = np.array(mask)
            root_dir = self.root_dir.split('/')
            new_dir = ''
            for dir in root_dir[:-2]:
                new_dir += dir + '/'

            img = Image.open(os.path.join(new_dir, 'images', img_name)).convert('RGB')

            mask = Image.open(os.path.join(new_dir, 'labels_instance', img_name[:-4] + '_label.png'))

            sample = [img, mask]
            sample = self.transform(sample)

        return sample, str(img_name[:-4])

    def __len__(self):
        return self.num_samples
