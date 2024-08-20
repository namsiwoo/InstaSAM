import os, argparse
import numpy as np
from PIL import Image
import skimage.io as io
from skimage.exposure import rescale_intensity
from utils.utils import mk_colored
from utils.hv_process import make_instance_sonnet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--img', action='store_true')
    parser.add_argument('--label', action='store_true')
    parser.add_argument('--label_vis', action='store_true')
    parser.add_argument('--num_img', default=5, type=int)
    parser.add_argument('--cell', action='store_true')
    parser.add_argument('--nuclei', action='store_true')
    parser.add_argument('--point', action='store_true')

    args = parser.parse_args()


    ### Load the test split
    data_path = '/media/NAS/nas_70/open_dataset/DeepLIIF'
    train_path = 'DeepLIIF_Training_Set'
    val_path = 'DeepLIIF_Validation_Set'
    test_path = 'DeepLIIF_Testing_Set'

    img_classes = ['IHC', 'Hematoxylin', 'DAPI', 'Lap2', 'Ki67','masks']
    img_size = 512

    dict_list = []
    print('train.....')
    if args.train == True:
        img_list = os.listdir(os.path.join(data_path, train_path))
        for i in range(len(img_classes)):
            os.makedirs(os.path.join(data_path, 'DeepLIIF', img_classes[i], 'train'), exist_ok=True)
        os.makedirs(os.path.join(data_path, 'DeepLIIF', 'positive_mask', 'train'), exist_ok=True)
        os.makedirs(os.path.join(data_path, 'DeepLIIF', 'negative_mask', 'train'), exist_ok=True)
        os.makedirs(os.path.join(data_path, 'DeepLIIF','labels_instance', 'train'), exist_ok=True)
        for img_name in img_list:
            img = np.array(Image.open(os.path.join(data_path, train_path, img_name)))
            for i in range(len(img_classes)):
                crop_img = img[:, i*img_size: (i+1)*img_size, :]
                if i == 5:
                    positive = (crop_img[:, :, 0] != 0).astype(np.uint8)*255
                    negative = (crop_img[:, :, 2] != 0).astype(np.uint8)*255
                    instance = (np.sum(crop_img, axis=2)>0).astype(np.uint8)
                    instance, _, _ = make_instance_sonnet(instance, positive+negative)
                    positive = Image.fromarray(positive.astype('uint8')).convert('L')
                    negative = Image.fromarray(negative.astype('uint8')).convert('L')
                    instance = Image.fromarray(instance.astype('uint16'))

                    positive.save(os.path.join(data_path, 'DeepLIIF', 'positive_mask', 'train', img_name))
                    negative.save(os.path.join(data_path, 'DeepLIIF', 'negative_mask', 'train', img_name))
                    instance.save(os.path.join(data_path, 'DeepLIIF', 'labels_instance', 'train', img_name))

                crop_img = Image.fromarray(crop_img.astype(np.uint8))
                crop_img.save(os.path.join(data_path, 'DeepLIIF', img_classes[i], 'train', img_name))

    print('val.....')
    if args.val == True:
        img_list = os.listdir(os.path.join(data_path, val_path))
        for i in range(len(img_classes)):
            os.makedirs(os.path.join(data_path, 'DeepLIIF', img_classes[i], 'val'), exist_ok=True)
        os.makedirs(os.path.join(data_path, 'DeepLIIF', 'positive_mask', 'val'), exist_ok=True)
        os.makedirs(os.path.join(data_path, 'DeepLIIF', 'negative_mask', 'val'), exist_ok=True)
        os.makedirs(os.path.join(data_path, 'DeepLIIF','labels_instance', 'val'), exist_ok=True)
        for img_name in img_list:
            img = np.array(Image.open(os.path.join(data_path, val_path, img_name)))
            for i in range(len(img_classes)):
                crop_img = img[:, i * img_size: (i + 1) * img_size, :]
                if i == 5:
                    positive = (crop_img[:, :, 0] == 1).astype(np.uint8)*255
                    negative = (crop_img[:, :, 2] == 1).astype(np.uint8)*255
                    instance = (np.sum(crop_img, axis=2)>0).astype(np.uint8)
                    instance, _, _ = make_instance_sonnet(instance, positive+negative)

                    positive = Image.fromarray(positive.astype('uint8')).convert('L')
                    negative = Image.fromarray(negative.astype('uint8')).convert('L')
                    instance = Image.fromarray(instance.astype('uint16'))

                    positive.save(os.path.join(data_path, 'DeepLIIF', 'positive_mask', 'val', img_name))
                    negative.save(os.path.join(data_path, 'DeepLIIF', 'negative_mask', 'val', img_name))
                    instance.save(os.path.join(data_path, 'DeepLIIF', 'labels_instance', 'val', img_name))
                crop_img = Image.fromarray(crop_img.astype(np.uint8))
                crop_img.save(os.path.join(data_path, 'DeepLIIF', img_classes[i], 'val', img_name))

    print('test.....')
    if args.test == True:
        img_list = os.listdir(os.path.join(data_path, test_path))
        for i in range(len(img_classes)):
            os.makedirs(os.path.join(data_path, 'DeepLIIF', img_classes[i], 'test'), exist_ok=True)
        os.makedirs(os.path.join(data_path, 'DeepLIIF', 'positive_mask', 'test'), exist_ok=True)
        os.makedirs(os.path.join(data_path, 'DeepLIIF', 'negative_mask', 'test'), exist_ok=True)
        os.makedirs(os.path.join(data_path, 'DeepLIIF','labels_instance', 'test'), exist_ok=True)
        for img_name in img_list:
            img = np.array(Image.open(os.path.join(data_path, test_path, img_name)))
            for i in range(len(img_classes)):
                crop_img = img[:, i * img_size: (i + 1) * img_size, :]
                if i == 5:
                    positive = (crop_img[:, :, 0] == 1).astype(np.uint8)*255
                    negative = (crop_img[:, :, 2] == 1).astype(np.uint8)*255
                    instance = (np.sum(crop_img, axis=2)>0).astype(np.uint8)
                    instance, _, _ = make_instance_sonnet(instance, positive+negative)

                    positive = Image.fromarray(positive.astype('uint8')).convert('L')
                    negative = Image.fromarray(negative.astype('uint8')).convert('L')
                    instance = Image.fromarray(instance.astype('uint16'))

                    positive.save(os.path.join(data_path, 'DeepLIIF', 'positive_mask', 'test', img_name))
                    negative.save(os.path.join(data_path, 'DeepLIIF', 'negative_mask', 'test', img_name))
                    instance.save(os.path.join(data_path, 'DeepLIIF', 'labels_instance', 'test', img_name))
                crop_img = Image.fromarray(crop_img.astype(np.uint8))
                crop_img.save(os.path.join(data_path, 'DeepLIIF', img_classes[i], 'test', img_name))
