import os

import numpy as np
import skimage.io as io

### Load the test split

npz_dir = '/media/NAS/nas_70/open_dataset/DeepCell'
test_dict = np.load(os.path.join(npz_dir, 'tissuenet_v1.1_train.npz'))

### Get the image data from the npz

test_X, test_y = test_dict['X'], test_dict['y']

### Create overlays of image data and labels

print(test_X.shape)
print(test_y.shape)
