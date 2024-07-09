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


class DeepCell_dataset(torch.utils.data.Dataset): #MO, CPM, CoNSeP
    def __init__(self, args, split, use_mask=False, data=False):
        self.args = args
        self.root_dir = os.path.expanduser(self.args.data_path)  # /media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg
        self.split = split
        self.use_mask = use_mask
        self.data=data

        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

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
        self.samples_X, self.samples_y = self.read_samples(self.root_dir, self.split, few_shot=args.fs)

        # set num samples
        self.num_samples = len(self.samples_X)
        print('{} dataset {} loaded'.format(self.split, self.num_samples))


    def read_samples(self, root_dir, split, few_shot=False):
        samples = np.load(os.path.join(self.root_dir, 'tissuenet_v1.1_{}.npz'.format(split)))
        samples_X, samples_y = samples['X'], samples['y']
        return samples_X, samples_y

    def create_rgb_image(self, input_data, channel_colors):
        from skimage.exposure import rescale_intensity
        """Takes a stack of 1- or 2-channel data and converts it to an RGB image

        Args:
            input_data: 4D stack of images to be converted to RGB
            channel_colors: list specifying the color for each channel

        Returns:
            numpy.array: transformed version of input data into RGB version

        Raises:
            ValueError: if ``len(channel_colors)`` is not equal
                to number of channels
            ValueError: if invalid ``channel_colors`` provided
            ValueError: if input_data is not 4D, with 1 or 2 channels
        """

        if len(input_data.shape) != 4:
            raise ValueError('Input data must be 4D, '
                             f'but provided data has shape {input_data.shape}')

        if input_data.shape[3] > 2:
            raise ValueError('Input data must have 1 or 2 channels, '
                             f'but {input_data.shape[-1]} channels were provided')

        valid_channels = ['red', 'green', 'blue']
        channel_colors = [x.lower() for x in channel_colors]

        if not np.all(np.isin(channel_colors, valid_channels)):
            raise ValueError('Only red, green, or blue are valid channel colors')

        if len(channel_colors) != input_data.shape[-1]:
            raise ValueError('Must provide same number of channel_colors as channels in input_data')

        rgb_data = np.zeros(input_data.shape[:3] + (3,), dtype='float32')

        # rescale channels to aid plotting
        for img in range(input_data.shape[0]):
            for channel in range(input_data.shape[-1]):
                current_img = input_data[img, :, :, channel]
                non_zero_vals = current_img[np.nonzero(current_img)]

                # if there are non-zero pixels in current channel, we rescale
                if len(non_zero_vals) > 0:
                    percentiles = np.percentile(non_zero_vals, [5, 95])
                    rescaled_intensity = rescale_intensity(current_img,
                                                           in_range=(0, percentiles[1]),
                                                           out_range=(0, 255))

                    # get rgb index of current channel
                    color_idx = np.where(np.isin(valid_channels, channel_colors[channel]))
                    rgb_data[img, :, :, color_idx] = rescaled_intensity

        # create a blank array for red channel
        return rgb_data
    def __getitem__(self, index):
        img_name = str(index)
        img = np.expand_dims(self.samples_X[index], 0)
        img = self.create_rgb_image(img, ['green', 'blue'])
        img = Image.fromarray(img[0].astype(np.uint8)).convert('RGB')

        box_label = skimage.morphology.label(self.samples_y[index])
        box_label = Image.fromarray(box_label.astype(np.uint16))

        sample = [img, box_label]

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
            # mask = Image.open(os.path.join(new_dir, 'labels_instance', self.split, img_name)) #pannuke
            # mask = Image.open(os.path.join(new_dir, 'labels_instance', self.split, img_name[:-4] + '_label.png'))
            mask = Image.open(os.path.join(new_dir, 'labels_instance', self.split, img_name[:-8] + '_masks.png')) #cellpose

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
