import os
import numpy as np
from PIL import Image
import skimage.io as io
from skimage.exposure import rescale_intensity
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

                # percentiles = np.percentile(non_zero_vals, [5, 95])
                rescaled_intensity = rescale_intensity(current_img,
                                                       in_range=(0, 255),)

                # get rgb index of current channel
                color_idx = np.where(np.isin(valid_channels, channel_colors[channel]))
                rgb_data[img, :, :, color_idx] = rescaled_intensity

    # create a blank array for red channel
    return rgb_data

### Load the test split

npz_dir = '/media/NAS/nas_70/open_dataset/DeepCell'
train_dict = np.load(os.path.join(npz_dir, 'tissuenet_v1.1_train.npz'))
val_dict = np.load(os.path.join(npz_dir, 'tissuenet_v1.1_val.npz'))
test_dict = np.load(os.path.join(npz_dir, 'tissuenet_v1.1_test.npz'))
### Get the image data from the npz

train_X, train_y = test_dict['X'], test_dict['y']
val_X, val_y = test_dict['X'], test_dict['y']
test_X, test_y = test_dict['X'], test_dict['y']

### Create overlays of image data and labels
train_x = create_rgb_image(train_X, ['green', 'blue'])
val_X = create_rgb_image(val_X, ['green', 'blue'])
test_X = create_rgb_image(test_X, ['green', 'blue'])

# os.makedirs(os.path.join(npz_dir, 'images', 'train'))
# os.makedirs(os.path.join(npz_dir, 'labels_instance', 'train'))
for i in range(5): #train_X.shape[0]
    img = train_X[i]
    img = Image.fromarray(img).convert('RGB')
    img.save(os.path.join(npz_dir, 'images', 'train', str(i)+'.png'))

    print(i, np.unique(train_y[i]))



print(train_X.shape, train_y.shape)
print(np.unique(train_X)[0], np.unique(train_X)[-1])
print(np.unique(train_y))
print(val_X.shape, val_y.shape)
print(test_X.shape, test_y.shape)

