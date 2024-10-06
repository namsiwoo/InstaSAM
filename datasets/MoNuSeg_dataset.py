import os, json
import numpy as np
import torch.utils.data

from PIL import Image

from scipy.ndimage.morphology import binary_dilation
import skimage.morphology, skimage.measure

from datasets.get_transforms_ssl import get_transforms

class Weak_dataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        self.args = args
        self.root_dir = os.path.expanduser(self.args.data_path)  # /media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg
        self.split = split

        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

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

            point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name[:-4] + '_label_point.png')).convert('L')
            point = binary_dilation(np.array(point), iterations=2)
            point = Image.fromarray(point)

            sample = [img, point]#, cluster_label, voronoi_label]  # , new_mask
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






