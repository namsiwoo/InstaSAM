import os

import numpy as np
from PIL import Image
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
train_x = np.zeros((train_X.shape[0], train_X.shape[1], train_X.shape[2], 1))
val_x = np.zeros((val_X.shape[0], val_X.shape[1], val_X.shape[2], 1))
test_x = np.zeros((test_X.shape[0], test_X.shape[1], test_X.shape[2], 1))

train_X = np.concatenate((train_X, train_x), axis=-1)
# os.makedirs(os.path.join(npz_dir, 'images', 'train'))
# os.makedirs(os.path.join(npz_dir, 'labels_instance', 'train'))
for i in range(train_X.shape[0], 5):
    img = train_X[i]
    Image.fromarray(img.astype(np.uint8))
    Image.save(os.path.join(npz_dir, 'images', 'train', str(i)+'.png'))


val_X = np.concatenate((val_X, val_x), axis=-1)
# os.makedirs(os.path.join(npz_dir, 'images', 'val'))
# os.makedirs(os.path.join(npz_dir, 'labels_instance', 'val'))

test_X = np.concatenate((test_X, test_x), axis=-1)
# os.makedirs(os.path.join(npz_dir, 'images', 'test'))
# os.makedirs(os.path.join(npz_dir, 'labels_instance', 'test'))


print(train_X.shape, train_y.shape)
print(np.unique(train_X)[0], np.unique(train_X)[-1])
print(np.unique(train_y))
print(val_X.shape, val_y.shape)
print(test_X.shape, test_y.shape)

