import os

import numpy as np
import skimage.io as io

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

print(train_X.shape, train_y.shape)
print(val_X.shape, val_y.shape)
print(test_X.shape, test_y.shape)

