import os, argparse
import numpy as np
from PIL import Image
import skimage.io as io
from skimage.exposure import rescale_intensity
from utils.utils import mk_colored

def create_rgb_image(input_data, channel_colors):
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
                                                       in_range=(percentiles[0], percentiles[1]),
                                                       out_range=(0, 255))
                # rescaled_intensity = rescale_intensity(current_img, out_range=(0, 255))

                # get rgb index of current channel
                color_idx = np.where(np.isin(valid_channels, channel_colors[channel]))
                rgb_data[img, :, :, color_idx] = rescaled_intensity

    # create a blank array for red channel
    return rgb_data


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
    npz_dir = '/media/NAS/nas_70/open_dataset/DeepCell'

    dict_list = []
    if args.train == True:
        train_dict = np.load(os.path.join(npz_dir, 'tissuenet_v1.1_train.npz'))
        train_X, train_y = train_dict['X'], train_dict['y']
        train_X = create_rgb_image(train_X, ['green', 'blue'])
        dict_list.append((train_X, train_y, 'train'))

    if args.val == True:
        val_dict = np.load(os.path.join(npz_dir, 'tissuenet_v1.1_val.npz'))
        val_X, val_y = val_dict['X'], val_dict['y']
        val_X = create_rgb_image(val_X, ['green', 'blue'])
        dict_list.append((val_X, val_y, 'val'))


    if args.test == True:
        test_dict = np.load(os.path.join(npz_dir, 'tissuenet_v1.1_test.npz'))
        test_X, test_y = test_dict['X'], test_dict['y']
        test_X = create_rgb_image(test_X, ['green', 'blue'])
        dict_list.append((test_X, test_y, 'test'))

    for i in range(len(dict_list)):
        X, y_label, split = dict_list[i]
        print(split, 'ing....')
        if args.num_img == 0:
            num_img = len(X)
        else:
            num_img = args.num_img

        for idx in range(num_img):
            img, label_ori = X[idx], y_label[idx]
            if args.img == True:
                img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
                img.save(os.path.join(npz_dir, 'images', split, str(idx) + '.png'))

            if args.label == True:
                if args.cell == True:
                    label = label_ori[:, :, 0]
                    if args.label_vis ==True:
                        label = mk_colored(label)*255
                        label = label.astype(np.uint8)
                        label = Image.fromarray(label).convert('RGB')
                        img_name = str(idx)+'_vis.png'
                    else:
                        label = label.astype(np.uint16)
                        img_name = str(idx)+'.png'
                        if args.point == True:
                            point = np.zeros_like(label)
                            for ins in np.unique(label)[1:]:
                                coor = np.where(label==ins)
                                y, x = coor
                                point[round(np.mean(y)), round(np.mean(x))] = 255
                            point = Image.fromarray(point.astype(np.uint8))
                            point.save(os.path.join(npz_dir, 'labels_point_cell', split, img_name))

                        label = Image.fromarray(label)
                    patch_folder = 'labels_instance_cell'

                    label.save(os.path.join(npz_dir, patch_folder, split, img_name))
                if args.nuclei == True:
                    label = label_ori[:, :, 1]
                    if args.label_vis ==True:
                        label = mk_colored(label)*255
                        label = label.astype(np.uint8)
                        label = Image.fromarray(label).convert('RGB')
                        img_name = str(idx)+'_vis.png'
                    else:
                        label = label.astype(np.uint16)
                        img_name = str(idx)+'.png'
                        if args.point == True:
                            point = np.zeros_like(label)
                            for ins in np.unique(label)[1:]:
                                coor = np.where(label==ins)
                                y, x = coor
                                point[round(np.mean(y)), round(np.mean(x))] = 255
                            point = Image.fromarray(point.astype(np.uint8))
                            point.save(os.path.join(npz_dir, 'labels_point_nuclei', split, img_name))

                        label = Image.fromarray(label)
                    patch_folder = 'labels_instance_nuclei'
                    label.save(os.path.join(npz_dir, patch_folder, split, img_name))
