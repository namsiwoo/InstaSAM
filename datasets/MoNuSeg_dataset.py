import json
import os, random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data

from PIL import Image
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from scipy.ndimage.morphology import binary_dilation
import skimage.morphology, skimage.measure

class SAM_gen_dataset(torch.utils.data.Dataset):
    def __init__(self, args, split, sup=False):
        self.args = args
        self.root_dir = os.path.expanduser(self.args.data_path)  # /media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg
        self.split = split
        self.sup = sup
        from datasets.get_transforms_ssl import get_transforms

        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

        # Imagenet norm?
        # self.mean = np.array([0.485,0.456,0.406])
        # self.std = np.array([0.229,0.224,0.225])

        # create image augmentation
        if self.split == 'train':
            self.transform = get_transforms({
                'random_resize': [0.8, 1.25],
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

        # samples = os.listdir(os.path.join(root_dir, 'images', split))
        return samples

    def __getitem__(self, index):
        img_name = self.samples[index % len(self.samples)]

        if self.split == 'train':
            # 1) read image
            img = Image.open(os.path.join(self.root_dir, 'images', self.split, img_name)).convert('RGB')

            if self.sup == True:

                box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4]+'_label.png')))
                # box_label = np.array(Image.open(os.path.join('/media/NAS/nas_187/siwoo/train', 'labels_instance', self.split, img_name[:-4]+'_label.png')))
                box_label = skimage.morphology.label(box_label)
                box_label = Image.fromarray(box_label.astype(np.uint16))

                # 3) do image augmentation
                sample = [img, box_label]  # , new_mask
                sample = self.transform(sample)
            else:
                # box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name)))
                # box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4]+'_label.png')))
                # box_label = skimage.morphology.label(box_label)
                # box_label = Image.fromarray(box_label.astype(np.uint16))

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

                # point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name)).convert('L')

                point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name[:-4] + '_label_point.png')).convert('L')
                point = binary_dilation(np.array(point), iterations=2)
                point = Image.fromarray(point)

                # box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4]+'_label.png')))
                # box_label = skimage.morphology.label(box_label)
                # point = Image.fromarray(box_label.astype(np.uint16))

                # cluster_label = Image.open(os.path.join(self.root_dir, 'labels_cluster', self.split, img_name[:-4]+'_label_cluster.png')).convert('RGB')
                # voronoi_label = Image.open(os.path.join(self.root_dir, 'labels_voronoi', self.split, img_name[:-4] + '_label_vor.png')).convert('RGB')
                # cluster_label = Image.open(os.path.join(self.root_dir, 'labels_geo_cluster', self.split, img_name[:-4]+'_label_geo_cluster.png')).convert('RGB')
                # voronoi_label = Image.open(os.path.join(self.root_dir, 'labels_geo_voronoi', self.split, img_name[:-4] + '_label_geo_vor.png')).convert('RGB')

                sample = [img, point]#, cluster_label, voronoi_label]  # , new_mask
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

            # img = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/images/val', img_name)).convert('RGB')
            # mask = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/labels_instance/val', img_name))
            # img = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC/images/val', img_name)).convert('RGB')
            # mask = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC/labels_instance/val2', img_name))


            sample = [img, mask]
            sample = self.transform(sample)

        return sample, str(img_name[:-4])

    def __len__(self):
        return self.num_samples

class DA_dataset(torch.utils.data.Dataset): #MO, CPM, CoNSeP
    def __init__(self, args, split, use_mask=False, data=('CPM', 'BC'), train_IHC=False):
        self.args = args
        self.split = split
        self.use_mask = use_mask
        self.data1=data[0]
        self.data2=data[1]
        if train_IHC == True:
            self.path1 = 'IHC'
            self.path2 = 'images'
            self.ext1 = '.png'
            self.ext2 = '_label.png'
        else:
            self.path2 = 'IHC'
            self.path1 = 'images'
            self.ext2 = '.png'
            self.ext1 = '_label.png'


        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

        # print('Norm is not used')
        # self.mean = np.array([1, 1, 1])
        # self.std = np.array([1, 1, 1])

        if self.args.sup == True:
            from datasets.get_transforms_ori import get_transforms
            n_mask = 0
        # create image augmentation
        else:
            from datasets.get_transforms_ssl import get_transforms
            n_mask =1

        if self.split == 'train':
            self.transform = get_transforms({
                # 'random_resize': [0.8, 1.25],
                'horizontal_flip': True,
                'random_affine': 0.3,
                'random_rotation': 90,
                'random_crop': 224,
                'label_encoding': [0, n_mask], #new_label: 3 else 2
                'to_tensor': 2, # number of img
                'normalize': np.array([self.mean, self.std])
            })
        else:
            self.transform = get_transforms({
                'to_tensor': 1,
                'normalize': np.array([self.mean, self.std])
            })

        # read samples
        self.samples = self.read_samples(self.split, few_shot=args.fs)

        # set num samples
        if self.split == 'train':
            self.num_samples = len(self.samples[0])
        else:
            self.num_samples = len(self.samples)
        print('{} dataset {} loaded'.format(self.split, self.num_samples))


    def read_samples(self, split, few_shot=False):
        samples2 = os.listdir(os.path.join(self.data2, self.path2, split))

        if split == 'train':
            samples1 = os.listdir(os.path.join(self.data1, self.path1, split))
            return samples1, samples2
        else:
            return samples2
    def __getitem__(self, index):
        if self.split == 'train':
            img_name = self.samples[1][random.randint(0, len(self.samples[1])-1)]
            img2 = Image.open(os.path.join(self.data2, self.path2, self.split, img_name)).convert('L').convert('RGB') #'RGB'

            img_name = self.samples[0][index % len(self.samples[0])]
            img1 = Image.open(os.path.join(self.data1, self.path1, self.split, img_name)).convert('L').convert('RGB') #'RGB'
            if self.use_mask == True:
                box_label = np.array(Image.open(os.path.join(self.data1, 'labels_instance', self.split, img_name[:-4]+self.ext1)))
                box_label = skimage.morphology.label(box_label)
                box_label = Image.fromarray(box_label.astype(np.uint16))
                sample = [img1, img2, box_label]
            else:
                point = Image.open(os.path.join(self.data1, 'labels_point', self.split, img_name[:-4]+self.ext1)).convert('L')
                point = binary_dilation(np.array(point), iterations=2)
                point = Image.fromarray(point)
                sample = [img1, img2, point]
        else:
            img_name = self.samples[index % len(self.samples)]
            img2 = Image.open(os.path.join(self.data2, self.path2, self.split, img_name)).convert('L').convert('RGB') #'RGB'

            mask = Image.open(os.path.join(self.data2, 'labels_instance', self.split, img_name[:-4]+self.ext2))
            sample = [img2, mask]
        sample = self.transform(sample)

        return sample, str(img_name)
    def __len__(self):
        return self.num_samples

class IHC_dataset(torch.utils.data.Dataset): #MO, CPM, CoNSeP
    def __init__(self, args, split, use_mask=False, data='DAPI'):
        self.args = args
        self.root_dir = os.path.expanduser(self.args.data_path)  # /media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg
        self.split = split
        self.use_mask = use_mask
        self.data=data

        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

        # print('Norm is not used')
        # self.mean = np.array([1, 1, 1])
        # self.std = np.array([1, 1, 1])

        if self.args.sup == True:
            from datasets.get_transforms_ori import get_transforms
            n_mask = 0
        # create image augmentation
        else:
            from datasets.get_transforms_ssl import get_transforms
            n_mask =1

        if self.split == 'train':
            self.transform = get_transforms({
                # 'random_resize': [0.8, 1.25],
                'horizontal_flip': True,
                'random_affine': 0.3,
                'random_rotation': 90,
                'random_crop': 224,
                'label_encoding': [0, n_mask], #new_label: 3 else 2
                'to_tensor': 1, # number of img
                'normalize': np.array([self.mean, self.std])
            })
        else:
            self.transform = get_transforms({
                'to_tensor': 1,
                'normalize': np.array([self.mean, self.std])
            })

        # read samples
        self.samples = self.read_samples(self.root_dir, self.split, few_shot=args.fs)

        # set num samples
        self.num_samples = len(self.samples)
        print('{} dataset {} loaded'.format(self.split, self.num_samples))


    def read_samples(self, root_dir, split, few_shot=False):
        samples = os.listdir(os.path.join(root_dir, self.data, split))
        return samples
    def __getitem__(self, index):
        img_name = self.samples[index % len(self.samples)]
        img = Image.open(os.path.join(self.root_dir, self.data, self.split, img_name)).convert('L').convert('RGB')

        if self.split == 'train':
            if self.use_mask == True:
                box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name)))
                box_label = skimage.morphology.label(box_label)
                box_label = Image.fromarray(box_label.astype(np.uint16))
                sample = [img, box_label]
            else:
                point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name)).convert('L')
                point = binary_dilation(np.array(point), iterations=2)
                point = Image.fromarray(point)
                sample = [img, point]
        else:
            mask = Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-3]+'png'))
            sample = [img, mask]
        sample = self.transform(sample)

        return sample, str(img_name)
    def __len__(self):
        return self.num_samples
class gt_with_weak_dataset(torch.utils.data.Dataset):
    def __init__(self, args, split, semi=False):
        self.args = args
        self.root_dir = os.path.expanduser(self.args.data_path)  # /media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg
        self.split = split
        self.semi = semi
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
                'random_resize': [0.8, 1.25],
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
        self.samples = self.read_samples(self.root_dir, self.split)

        # self.samples = self.read_samples('/media/NAS/nas_70/open_dataset/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg', self.split)
        # self.samples += self.read_samples('/media/NAS/nas_70/open_dataset/CPM/CPM 17/via instance learning data_for_train/CPM 17', self.split)
        # self.samples += self.read_samples('/media/NAS/nas_32/siwoo/TNBC/TNBC/via instance learning data_for_train/TNBC', self.split)

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
            img = Image.open(os.path.join(self.root_dir, 'images', self.split, img_name)).convert('RGB')

            if self.semi == False:
                point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name[:-4] + '_label_point.png')).convert('L')
                point = binary_dilation(np.array(point), iterations=2)
                point = Image.fromarray(point)

                box_label = Image.open(os.path.join('/media/NAS/nas_70/siwoo_data/UDA_citycapes/CoNSeP/masks', img_name))
                # box_label = Image.open(os.path.join('/media/NAS/nas_70/siwoo_data/UDA_citycapes/TNBC/masks', img_name))

                # 3) do image augmentation
                sample = [img, box_label, point]  # , new_mask
                sample = self.transform(sample)
            else:
                point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name[:-4] + '_label_point.png')).convert('L')
                point = binary_dilation(np.array(point), iterations=2)
                point = Image.fromarray(point)


                # if img_name[:-5] == '7':
                # if img_name[-7:-5] == '2_3': #tnbc
                # if img_name[-6:-4] == '_3' or img_name[-6:-4] == '11':
                if img_name[-6:-4] == '_3':  # CPM
                    box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4]+'_label.png')))
                    box_label = skimage.morphology.label(box_label)
                    box_label = Image.fromarray(box_label.astype(np.uint16))
                else:
                    box_label = np.zeros_like(np.array(point))
                    box_label = Image.fromarray(box_label.astype(np.uint16))

                sample = [img, box_label, point]#, cluster_label, voronoi_label]  # , new_mask
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

            # img = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/images/val', img_name)).convert('RGB')
            # mask = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/labels_instance/val', img_name))
            # img = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC/images/val', img_name)).convert('RGB')
            # mask = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC/labels_instance/val2', img_name))


            sample = [img, mask]
            sample = self.transform(sample)

        return sample, str(img_name[:-4])

    def __len__(self):
        return self.num_samples
class Galaxy_dataset(torch.utils.data.Dataset): #MO, CPM, CoNSeP
    def __init__(self, args, split, use_mask=False, data='nuclei'):
        self.args = args
        self.root_dir = os.path.expanduser(self.args.data_path)  # /media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg
        self.split = split
        self.use_mask = use_mask
        self.data=data

        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        # self.mean = np.array([1, 1, 1])
        # self.std = np.array([1, 1, 1])

        if self.args.sup == True:
            from datasets.get_transforms_ori import get_transforms
            n_mask = 0
        # create image augmentation
        else:
            from datasets.get_transforms_ssl import get_transforms
            n_mask =1

        if self.split == 'train':
            self.transform = get_transforms({
                # 'random_resize': [0.8, 1.25],
                'horizontal_flip': True,
                'random_affine': 0.3,
                'random_rotation': 90,
                'random_crop': 224,
                'label_encoding': [0, n_mask], #new_label: 3 else 2
                'to_tensor': 1, # number of img
                'normalize': np.array([self.mean, self.std])
            })
        else:
            self.transform = get_transforms({
                'to_tensor': 1,
                'normalize': np.array([self.mean, self.std])
            })

        # read samples
        self.samples = self.read_samples(self.root_dir, self.split, few_shot=args.fs)

        # set num samples
        self.num_samples = len(self.samples)
        print('{} dataset {} loaded'.format(self.split, self.num_samples))


    def read_samples(self, root_dir, split, few_shot=False):
        samples = os.listdir(os.path.join(root_dir, split, 'images'))
        return samples
    def __getitem__(self, index):
        img_name = self.samples[index % len(self.samples)]
        # img = Image.open(os.path.join(self.root_dir, 'images', self.split, img_name)).convert('RGB')
        img = np.load(os.path.join(self.root_dir, self.split, 'images', img_name))*255
        img = Image.fromarray(img.astype(np.uint8))

        if self.split == 'train':
            if self.args.sup == True:
                box_label = np.load(os.path.join(self.root_dir, self.split, 'masks', img_name))
                box_label = skimage.morphology.label(box_label)
                box_label = Image.fromarray(box_label.astype(np.uint16))
                sample = [img, box_label]
            else:
                point = np.load(os.path.join(self.root_dir, self.split, 'points', img_name))
                point = binary_dilation(np.array(point), iterations=2)
                point = Image.fromarray(point).convert('L')
                sample = [img, point]
            # else:
            #     if self.data == 'nuclei':
            #         point = Image.open(os.path.join(self.root_dir, 'labels_point_nuclei', self.split, img_name)).convert('L')
            #     else:
            #         point = Image.open(os.path.join(self.root_dir, 'labels_point_cell', self.split, img_name)).convert('L')
            #     point = binary_dilation(np.array(point), iterations=2)
            #     point = Image.fromarray(point)
            #     sample = [img, point]
        else:
            # box_label = np.load(os.path.join(self.root_dir, self.split, 'masks', img_name))
            box_label = np.array(Image.open((os.path.join(self.root_dir, self.split, 'vis', img_name[:-3]+'png'))).convert('L'))

            box_label = skimage.morphology.label(box_label)
            box_label = Image.fromarray(box_label.astype(np.uint16))
            sample = [img, box_label]
        sample = self.transform(sample)

        return sample, str(img_name)

    def __len__(self):
        return self.num_samples

class DeepCell_dataset(torch.utils.data.Dataset): #MO, CPM, CoNSeP
    def __init__(self, args, split, use_mask=False, data='nuclei'):
        self.args = args
        self.root_dir = os.path.expanduser(self.args.data_path)  # /media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg
        self.split = split
        self.use_mask = use_mask
        self.data=data

        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

        # print('Norm is not used')
        # self.mean = np.array([1, 1, 1])
        # self.std = np.array([1, 1, 1])

        if self.args.sup == True:
            from datasets.get_transforms_ori import get_transforms
            n_mask = 0
        # create image augmentation
        else:
            from datasets.get_transforms_ssl import get_transforms
            n_mask =1

        if self.split == 'train':
            self.transform = get_transforms({
                # 'random_resize': [0.8, 1.25],
                'horizontal_flip': True,
                'random_affine': 0.3,
                'random_rotation': 90,
                'random_crop': 224,
                'label_encoding': [0, n_mask], #new_label: 3 else 2
                'to_tensor': 1, # number of img
                'normalize': np.array([self.mean, self.std])
            })
        else:
            self.transform = get_transforms({
                'to_tensor': 1,
                'normalize': np.array([self.mean, self.std])
            })

        # read samples
        self.samples = self.read_samples(self.root_dir, self.split, few_shot=args.fs)

        # set num samples
        self.num_samples = len(self.samples)
        print('{} dataset {} loaded'.format(self.split, self.num_samples))


    def read_samples(self, root_dir, split, few_shot=False):
        samples = os.listdir(os.path.join(root_dir, 'images', split))
        return samples
    def __getitem__(self, index):
        img_name = self.samples[index % len(self.samples)]
        img = Image.open(os.path.join(self.root_dir, 'images', self.split, img_name)).convert('RGB')

        if self.split == 'train':
            if self.use_mask == True:
                if self.data == 'nuclei':
                    box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance_nuclei', self.split, img_name)))
                else:
                    box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance_cell', self.split, img_name)))
                box_label = skimage.morphology.label(box_label)
                box_label = Image.fromarray(box_label.astype(np.uint16))


                # plt.imshow(box_label)
                # plt.colorbar()
                # print(np.unique(np.array(box_label)))
                # plt.savefig(os.path.join(self.args.result, 'img/0/'+str(index)+'.png'))

                sample = [img, box_label]
            else:
                if self.data == 'nuclei':
                    point = Image.open(os.path.join(self.root_dir, 'labels_point_nuclei', self.split, img_name)).convert('L')
                else:
                    point = Image.open(os.path.join(self.root_dir, 'labels_point_cell', self.split, img_name)).convert('L')
                point = binary_dilation(np.array(point), iterations=2)
                point = Image.fromarray(point)
                sample = [img, point]
        else:
            if self.data == 'nuclei':
                mask = Image.open(os.path.join(self.root_dir, 'labels_instance_nuclei', self.split, img_name[:-3]+'png'))
            else:
                mask = Image.open(os.path.join(self.root_dir, 'labels_instance_cell', self.split, img_name[:-3]+'png'))

            sample = [img, mask]
        sample = self.transform(sample)

        return sample, str(img_name)

    def __len__(self):
        return self.num_samples

class Crop_dataset(torch.utils.data.Dataset): #MO, CPM, CoNSeP
    def __init__(self, args, split, use_mask=False, data=False):
        self.args = args
        self.root_dir = os.path.expanduser(self.args.data_path)  # /media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg
        self.split = split
        self.use_mask = use_mask
        self.data=data

        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

        # print('Norm is not used')
        # self.mean = np.array([1, 1, 1])
        # self.std = np.array([1, 1, 1])

        if self.args.sup == True:
            from datasets.get_transforms_ori import get_transforms
            n_mask = 0
        # create image augmentation
        else:
            from datasets.get_transforms_ssl import get_transforms
            n_mask =1

        if self.split == 'train':
            self.transform = get_transforms({
                # 'random_resize': [0.8, 1.25],
                'horizontal_flip': True,
                'random_affine': 0.3,
                'random_rotation': 90,
                'random_crop': 224,
                'label_encoding': [0, n_mask], #new_label: 3 else 2
                'to_tensor': 1, # number of img
                'normalize': np.array([self.mean, self.std])
            })
        else:
            self.transform = get_transforms({
                'to_tensor': 1,
                'normalize': np.array([self.mean, self.std])
            })

        # read samples
        self.samples = self.read_samples(self.root_dir, self.split, few_shot=args.fs)

        # set num samples
        self.num_samples = len(self.samples)
        print('{} dataset {} loaded'.format(self.split, self.num_samples))

    def read_samples(self, root_dir, split, few_shot=False):
        if self.data == 'pannuke' or 'cellpose':
            samples = os.listdir(os.path.join(root_dir, 'images', split))
        else:
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
        return samples

    def __getitem__(self, index):
        img_name = self.samples[index % len(self.samples)]

        if self.split == 'train':
            # 1) read image
            img = Image.open(os.path.join(self.root_dir, 'images', self.split, img_name)).convert('RGB')



            if self.use_mask == True:
                if self.data == 'pannuke':
                    box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name)))
                elif self.data == 'cellpose':
                    box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-8]+'_masks.png')))
                else:
                    box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4] + '_label.png')))

                box_label = skimage.morphology.label(box_label)
                box_label = Image.fromarray(box_label.astype(np.uint16))

                sample = [img, box_label]#, cluster_label, voronoi_label]  # , new_mask
            else:
                if self.data == 'pannuke':
                    point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name)).convert('L')
                else:
                    point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name[:-8] + '_labels_point.png')).convert('L')
                point = binary_dilation(np.array(point), iterations=2)
                point = Image.fromarray(point)


                # box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-8]+'_masks.png')))
                #
                # box_label = skimage.morphology.label(box_label)
                # box_label = Image.fromarray(box_label.astype(np.uint16))

                sample = [img, point]
            sample = self.transform(sample)

        else:
            if self.data == 'pannuke' or 'cellpose':
                new_dir = self.root_dir

            else:
                root_dir = self.root_dir.split('/')
                new_dir = ''
                for dir in root_dir[:-2]:
                    new_dir += dir + '/'

            img = Image.open(os.path.join(new_dir, 'images', self.split, img_name)).convert('RGB')
            if self.data == 'pannuke':
                mask = Image.open(os.path.join(new_dir, 'labels_instance', self.split, img_name)) #pannuke
            # mask = Image.open(os.path.join(new_dir, 'labels_instance', self.split, img_name[:-4] + '_label.png'))
            else:
                mask = Image.open(os.path.join(new_dir, 'labels_instance', self.split, img_name[:-8] + '_masks.png')) #cellpose

            sample = [img, mask]
            sample = self.transform(sample)

        return sample, str(img_name[:-4])

    def __len__(self):
        return self.num_samples

class MoNuSeg_weak_dataset(torch.utils.data.Dataset):
    def __init__(self, args, split, sup=False):
        self.args = args
        self.root_dir = os.path.expanduser(self.args.data_path)  # /media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg
        self.split = split
        self.sup = sup
        if self.sup == True:
            from datasets.get_transforms_ori import get_transforms
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
                'random_resize': [0.8, 1.25],
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
        self.samples = self.read_samples(self.root_dir, self.split)

        # self.samples = self.read_samples('/media/NAS/nas_70/open_dataset/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg', self.split)
        # self.samples += self.read_samples('/media/NAS/nas_70/open_dataset/CPM/CPM 17/via instance learning data_for_train/CPM 17', self.split)
        # self.samples += self.read_samples('/media/NAS/nas_32/siwoo/TNBC/TNBC/via instance learning data_for_train/TNBC', self.split)

        # set num samples
        self.num_samples = len(self.samples)

        print('{} dataset {} loaded'.format(self.split, self.num_samples))

    def read_samples(self, root_dir, split):
        if split == 'train':
            samples = os.listdir(os.path.join(root_dir, 'images', split))
            # samples = os.listdir(os.path.join('/media/NAS/nas_70/open_dataset/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg', 'images', 'train_few_shot'))
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

            if self.sup == True:

                box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4]+'_label.png')))
                # box_label = np.array(Image.open(os.path.join('/media/NAS/nas_187/siwoo/train', 'labels_instance', self.split, img_name[:-4]+'_label.png')))
                box_label = skimage.morphology.label(box_label)
                box_label = Image.fromarray(box_label.astype(np.uint16))

                # 3) do image augmentation
                sample = [img, box_label]  # , new_mask
                sample = self.transform(sample)
            else:
                # box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name)))
                # box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4]+'_label.png')))
                # box_label = skimage.morphology.label(box_label)
                # box_label = Image.fromarray(box_label.astype(np.uint16))

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

                # point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name)).convert('L')

                point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name[:-4] + '_label_point.png')).convert('L')
                point = binary_dilation(np.array(point), iterations=2)
                point = Image.fromarray(point)

                # box_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4]+'_label.png')))
                # box_label = skimage.morphology.label(box_label)
                # point = Image.fromarray(box_label.astype(np.uint16))

                # cluster_label = Image.open(os.path.join(self.root_dir, 'labels_cluster', self.split, img_name[:-4]+'_label_cluster.png')).convert('RGB')
                # voronoi_label = Image.open(os.path.join(self.root_dir, 'labels_voronoi', self.split, img_name[:-4] + '_label_vor.png')).convert('RGB')
                # cluster_label = Image.open(os.path.join(self.root_dir, 'labels_geo_cluster', self.split, img_name[:-4]+'_label_geo_cluster.png')).convert('RGB')
                # voronoi_label = Image.open(os.path.join(self.root_dir, 'labels_geo_voronoi', self.split, img_name[:-4] + '_label_geo_vor.png')).convert('RGB')

                sample = [img, point]#, cluster_label, voronoi_label]  # , new_mask
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

            # img = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/images/val', img_name)).convert('RGB')
            # mask = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/labels_instance/val', img_name))
            # img = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC/images/val', img_name)).convert('RGB')
            # mask = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC/labels_instance/val2', img_name))


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

if __name__ == '__main__':
    # path = '/media/NAS/nas_187/datasets/galaxy_dataset_UNIST/train/masks'
    # name = os.listdir(path)
    # for n in name:
    #     mask = np.load(os.path.join(path, n))
    #     point = np.zeros_like(mask)
    #     label_regions = skimage.measure.regionprops(mask)
    #     for i, region in enumerate(label_regions):
    #         # point_coords.append([round(region.centroid[0]), round(region.centroid[1])])
    #         point[round(region.centroid[0]), round(region.centroid[1])] = 255
    #     np.save(os.path.join('/media/NAS/nas_187/datasets/galaxy_dataset_UNIST/train/point', n[:-4] + '.npy'), point)

    import json, random
    from collections import OrderedDict

    # data_dir = '/media/NAS/nas_70/open_dataset/TNBC/TNBC/via instance learning data_for_train/TNBC'
    data_dir = '/media/NAS/nas_70/open_dataset/TNBC/TNBC/images'
    json_dir = '/media/NAS/nas_70/open_dataset/TNBC/TNBC_new'


    data_list = os.listdir(data_dir)
    random.shuffle(data_list)

    train_list = data_list[:-7-13]
    val_list = data_list[-7-13:-13]
    test_list = data_list[-13:]


    # train_dir = os.path.join(data_dir, 'train', 'images')
    # train_list = os.listdir(train_dir)
    # val_list = ['image_01.png', 'image_05.png', 'image_07.png', 'image_13.png', 'image_20.png', 'image_30.png']
    # for i in val_list:
    #     train_list.remove(i)
    #
    # for i in train_list:
    #     img = Image.open(f"{train_dir}/{i}")
    #     img.save(f"{save_dir}/{i}")
    # for i in val_list:
    #     img = Image.open(f"{train_dir}/{i}")
    #     img.save(f"{save_dir}/{i}")
    #
    # test_dir = os.path.join(data_dir, 'test', 'images')
    # test_list = os.listdir(test_dir)
    # new_test_list = []
    # for i in range(len(test_list)):
    #     img = Image.open(f"{test_dir}/image_{str(i).zfill(2)}.png")
    #     img.save(f"{save_dir}/image_{str(i + len(train_list) + len(val_list)).zfill(2)}.png")
    #     new_test_list.append(f"image_{str(i + len(train_list) + len(val_list)).zfill(2)}.png")

    json_data = OrderedDict()
    json_data['train'] = train_list
    json_data['val'] = val_list
    json_data['test'] = test_list

    with open('{:s}/train_val_test.json'.format(json_dir), 'w') as make_file:
        json.dump(json_data, make_file, indent='\t')





