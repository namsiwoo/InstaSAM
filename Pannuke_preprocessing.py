import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure

def main_fold1():
    path = '/media/NAS/nas_70/open_dataset/pannuke/Fold 1'
    img = np.load(os.path.join('/media/NAS/nas_70/open_dataset/pannuke/Fold 1/images/fold1', 'images.npy'))
    mask = np.load(os.path.join('/media/NAS/nas_70/open_dataset/pannuke/Fold 1/masks/fold1', 'masks.npy'))

    for i in range(img.shape[0]):
        img1 = img[i]
        mask1 = mask[i]
        mask1 = np.sum(mask1, axis=-1)-1
        mask1 = measure.label(mask1)

        point = np.zeros_like(mask1)
        index = np.unique(mask1)

        for j in index[1:]:
            coor = np.where(mask1 == j)
            y, x = coor
            y, x = round(np.mean(y)), round(np.mean(x))
            point[y, x] = 255

        img1 = Image.fromarray(np.uint8(img1))
        mask1 = Image.fromarray(np.uint16(mask1))
        point1 = Image.fromarray(np.uint8(point))

        img1.save('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/images/train/'+str(i)+'.png')
        mask1.save('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/labels_instance/train/'+str(i)+'.png')
        point1.save('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/labels_point/train/'+str(i)+'.png')

def main_fold2():
    path = '/media/NAS/nas_70/open_dataset/pannuke/Fold 2/images/fold2'
    img = np.load(os.path.join('/media/NAS/nas_70/open_dataset/pannuke/Fold 2/images/fold2', 'images.npy'))
    mask = np.load(os.path.join('/media/NAS/nas_70/open_dataset/pannuke/Fold 2/masks/fold2', 'masks.npy'))
    print(img.shape)

    num_pixel = img.shape[0] * img.shape[1] * img.shape[2]
    total_sum = img.sum(axis=(0, 1, 2))
    total_square_sum = (img**2).sum(axis=(0, 1, 2))
    mean_values = total_sum / num_pixel
    std_values = np.sqrt(total_square_sum / num_pixel - mean_values ** 2)
    mean_values = mean_values / 255
    std_values = std_values / 255

    # np.save('{:s}/mean_std.npy'.format('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/val'), np.array([mean_values, std_values]))
    # np.savetxt('{:s}/mean_std.txt'.format('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/val'), np.array([mean_values, std_values]), '%.4f', '\t')

    for i in range(img.shape[0]):
        img1 = img[i]
        mask1 = mask[i]
        mask1 = np.sum(mask1, axis=-1)-1
        mask1 = measure.label(mask1)

        point = np.zeros_like(mask1)
        index = np.unique(mask1)

        for j in index[1:]:
            coor = np.where(mask1 == j)
            y, x = coor
            y, x = round(np.mean(y)), round(np.mean(x))
            point[y, x] = 255

        img1 = Image.fromarray(np.uint8(img1))
        mask1 = Image.fromarray(np.uint16(mask1))
        point1 = Image.fromarray(np.uint8(point))

        img1.save('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/images/val/'+str(i)+'.png')
        mask1.save('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/labels_instance/val/'+str(i)+'.png')
        point1.save('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/labels_point/val/'+str(i)+'.png')

def main_fold3():
    # path = '/media/NAS/nas_70/open_dataset/pannuke/Fold 2/images/fold3'
    img = np.load(os.path.join('/media/NAS/nas_70/open_dataset/pannuke/Fold 3/images/fold3', 'images.npy'))
    mask = np.load(os.path.join('/media/NAS/nas_70/open_dataset/pannuke/Fold 3/masks/fold3', 'masks.npy'))
    print(img.shape)

    num_pixel = img.shape[0] * img.shape[1] * img.shape[2]
    total_sum = img.sum(axis=(0, 1, 2))
    total_square_sum = (img**2).sum(axis=(0, 1, 2))
    mean_values = total_sum / num_pixel
    std_values = np.sqrt(total_square_sum / num_pixel - mean_values ** 2)
    mean_values = mean_values / 255
    std_values = std_values / 255

    # np.save('{:s}/mean_std.npy'.format('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/val'), np.array([mean_values, std_values]))
    # np.savetxt('{:s}/mean_std.txt'.format('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/val'), np.array([mean_values, std_values]), '%.4f', '\t')

    for i in range(img.shape[0]):
        img1 = img[i]
        mask1 = mask[i]
        mask1 = np.sum(mask1, axis=-1)-1
        mask1 = measure.label(mask1)

        point = np.zeros_like(mask1)
        index = np.unique(mask1)

        for j in index[1:]:
            coor = np.where(mask1 == j)
            y, x = coor
            y, x = round(np.mean(y)), round(np.mean(x))
            point[y, x] = 255

        img1 = Image.fromarray(np.uint8(img1))
        mask1 = Image.fromarray(np.uint16(mask1))
        point1 = Image.fromarray(np.uint8(point))

        img1.save('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/images/test/'+str(i)+'.png')
        mask1.save('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/labels_instance/test/'+str(i)+'.png')
        point1.save('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/labels_point/test/'+str(i)+'.png')



if __name__ == '__main__':
    # main_fold1()
    # main_fold2()
    # main_fold3()

    path = '/media/NAS/nas_70/open_dataset/pannuke/pannuke_for_cellvit/fold0/labels'
    for i in os.listdir(path):
        label = Image.open(os.path.join(path, i))
        label = np.array(label)

        npy_label = np.zeros_like(label)
        npy_label['']




    mask = np.array(Image.open('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/labels_instance/train/0.png'))
    print(mask.shape)
    print(np.unique(mask))
    plt.imshow(mask)
    plt.show()

    mask = np.array(Image.open('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/labels_instance/val/0.png'))
    print(mask.shape)
    print(np.unique(mask))
    plt.imshow(mask)
    plt.show()


