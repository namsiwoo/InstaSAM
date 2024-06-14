import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import skimage.morphology as ski_morph

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def mk_colored(instance_img):
    instance_img = instance_img.astype(np.int32)
    H, W = instance_img.shape[0], instance_img.shape[1]
    pred_colored_instance = np.zeros((H, W, 3))

    nuc_index = list(np.unique(instance_img))
    nuc_index.pop(0)

    for k in nuc_index:
        pred_colored_instance[instance_img == k, :] = np.array(get_random_color())

    return pred_colored_instance

def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random() for i in range(3)]
    return r, g, b

sam_checkpoint = "../sam_vit_h_4b8939.pth"
model_type = "vit_h"
# image_root = '/media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg/images/train'
# instance_root = '/media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg/labels_instance/train'
# point_root = '/media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg/labels_point/train'
# voronoi_root = '/media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg/labels_voronoi/train'
# mask_root = '/media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg/labels_instance/train'
#
# save_root = '/media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg/MoNuSeg/via instance learning data_for_train/MoNuSeg/labels_sam/train'

image_root = '/media/NAS/nas_187/PATHOLOGY_DATA/CoNSeP/CoNSeP/via instance learning data_for_train/CoNSeP/images/train'
instance_root = '/media/NAS/nas_187/PATHOLOGY_DATA/CoNSeP/CoNSeP/via instance learning data_for_train/CoNSeP/labels_instance/train'
point_root = '/media/NAS/nas_187/PATHOLOGY_DATA/CoNSeP/CoNSeP/via instance learning data_for_train/CoNSeP/labels_point/train'
voronoi_root = '/media/NAS/nas_187/PATHOLOGY_DATA/CoNSeP/CoNSeP/via instance learning data_for_train/CoNSeP/labels_voronoi/train'
mask_root = '/media/NAS/nas_187/PATHOLOGY_DATA/CoNSeP/CoNSeP/via instance learning data_for_train/CoNSeP/labels_instance/train'

save_root = '/media/NAS/nas_187/PATHOLOGY_DATA/CoNSeP/CoNSeP/via instance learning data_for_train/CoNSeP/labels_sam/train'



from segment_anything import SamPredictor, sam_model_registry

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

img_list = os.listdir(image_root)
img_list.sort()
point_list = os.listdir(point_root)
point_list.sort()
voronoi_list = os.listdir(voronoi_root)
voronoi_list.sort()
plt.figure(figsize=(10, 10))
for j in range(len(img_list)):
    img_list[0] = 'train_11_10.png'

    # image
    image = Image.open(os.path.join(image_root, img_list[j])).convert('RGB')
    image = np.array(image)

    mask = Image.open(os.path.join(instance_root, img_list[j][:-4] + '_label.png'))
    mask = np.array(mask)
    mask = ski_morph.label(mask)
    mask = mk_colored(mask)


    predictor.set_image(image)

    from scipy import ndimage

    # point to label
    point = Image.open(os.path.join(point_root, img_list[j][:-4] + '_label_point.png')).convert('L')
    point2 = np.array(point, dtype=float)  # [250,250], 가우시안 값 float으로 갖기 위해

    ## binaary dilation
    # point2 = ndimage.binary_dilation(point2, iterations = 8) #점 불리기

    ## gaussian filter using padding
    # point2 = np.pad(point2, 10, mode='mean')
    # point2 = ndimage.gaussian_filter(point2, sigma=3)
    # point2 = point2[10:260, 10:260]

    # point2[np.array(point)==255]=2.5

    point = np.array(point)
    point = np.where(point > 0)
    point_label = np.ones(len(point[0]), dtype=int)
    point_x = np.array(point[0])
    point_y = np.array(point[1])
    point_xy = []
    point_coords = []

    mask_image = np.zeros((250, 250))
    mask_image2 = np.zeros((250, 250))
    for i in range(len(point_x)):
        point_coords.append([point_y[i], point_x[i]])
    point_coords = np.array(point_coords)

    plt.subplot(1, 5, 1)
    plt.imshow(image)
    plt.subplot(1, 5, 2)
    plt.imshow(mask)
    plt.subplot(1, 5, 3)
    for i in range(len(point_x)):

        point_xy.append([point_y[i], point_x[i]])
        pc = np.array([[point_y[i], point_x[i]]])
        pl = np.array([1])
        masks, scores, logits = predictor.predict(point_labels=pl, point_coords=pc, multimask_output=False, )

        h, w = masks.shape[-2:]
        masks = masks.reshape(h, w)
        # if np.sum(masks!=0) > 250*250/6:
        #     masks[mask_image!=0] = 0
        #     mask_image[masks != 0] = 200
        # else:
        #     mask_image[masks!=0] = 10

        if scores[0] >0.8 and np.sum(masks!=0) < 250*250/6:
            mask_image2[masks!=0] = 10
        else:
            masks[mask_image2 != 0] = 0
            mask_image2[masks != 0] = 200

        point_labels = np.zeros(len(point[0]), dtype=int)
        point_labels[i] = 1
        masks, scores, logits = predictor.predict(point_labels=point_labels, point_coords=point_coords, multimask_output=False, )
        masks = masks.reshape(h, w)
        mask_image[masks != 0] = i+1

    point_xy = np.array(point_xy)
    # mask_image[mask_image>199] = 200
    mask_image2[mask_image2>199] = 200

    # sam_label = np.zeros((250, 250, 3))
    # sam_label[mask_image2==0, 0] = 255
    # sam_label[mask_image2==10, 1] = 255
    # sam_label = sam_label.astype(np.uint8)
    # sam_label = Image.fromarray(sam_label)
    # sam_label.save(os.path.join(save_root, img_list[j][:-4]+'_label_sam.png'))


    mask_image = mk_colored(mask_image)
    mask_image2 = mk_colored(mask_image2)
    plt.imshow(mask_image)
    show_points(point_xy, point_label, plt.gca())

    plt.subplot(1, 5, 4)
    plt.imshow(mask_image2)
    show_points(point_xy, point_label, plt.gca())

    masks, scores, logits = predictor.predict(point_labels=point_label, point_coords=point_xy, multimask_output=False, )
    plt.subplot(1, 5, 5)
    show_points(point_xy, point_label, plt.gca())
    show_mask(masks, plt.gca())
    plt.show()

    # point_xy = np.array(point_xy)
    # show_points(point_xy, point_label, plt.gca())
    # plt.show()


    # point_xy = np.array(point_xy)
    # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     #print(type(mask)) #numpy.ndarray
    #     plt.figure(figsize=(10,10))
    #     plt.imshow(image)
    #     show_mask(mask, plt.gca())
    #     show_points(cat_coords, cat_labels, plt.gca())
    #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #     plt.show()


# ---------------------------------------------------------------------------------------
    #voronoi line to point to label

    # voronoi = Image.open(os.path.join(voronoi_root, voronoi_list[0])).convert('RGB')
    # voronoi_line = np.array(voronoi) #[250,250,3]
    # voronoi_line = np.where(voronoi_line[..., 0] > 0)
    # voronoi_x = np.array(voronoi_line[0])
    # voronoi_y = np.array(voronoi_line[1])
    # voronoi_xy =[]
    # for i in range(len(voronoi_x)):
    #     voronoi_xy.append([voronoi_y[i],voronoi_x[i]])
    # voronoi_xy= np.array(voronoi_xy)
    # voronoi_xy = random.choices(voronoi_xy,k=20)
    # voronoi_label = np.zeros(20,dtype=int)
    #
    # cat_labels = np.concatenate((point_label, voronoi_label)) #[38,]
    # cat_coords = np.concatenate((point_xy, voronoi_xy)) #[38,2]
    ###############################


    # predictor.set_image(image)
    # masks, scores, logits = predictor.predict(point_labels= cat_labels,point_coords=cat_coords, multimask_output=True,)



    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # mask_generator = SamAutomaticMaskGenerator(model=sam,
    #                                            points_per_side=32,
    #                                            pred_iou_thresh=0.86,
    #                                            stability_score_thresh=0.92,
    #                                            crop_n_layers=2,
    #                                            crop_n_points_downscale_factor=2,
    #                                            min_mask_region_area=100,)
    # masks = mask_generator.generate(image)


    # print(masks)
    # print(masks.shape) #[3, 250, 250] (sam output 3 masks. h, w )
    # np.save('/home/sadiehong/Desktop/MoNuSeg/mask/mask{}'.format(j),masks) #save mask


    # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     #print(type(mask)) #numpy.ndarray
    #     plt.figure(figsize=(10,10))
    #     plt.imshow(image)
    #     show_mask(mask, plt.gca())
    #     show_points(cat_coords, cat_labels, plt.gca())
    #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #     plt.show()



"""
#mask retaining



mask_list = os.listdir(mask_root)
mask_list.sort()
#print(len(mask_list)) #3

for j in range(len(mask_list)):
    mask_init = np.load(os.path.join(mask_root, mask_list[j]))
    mask_init = mask_init.astype(int) #3,250,250
    print("maskinit:",mask_init.shape)
    #print("mask",mask.shape)

######################################
    #image
    image = Image.open(os.path.join(image_root, img_list[j])).convert('RGB')
    image = np.array(image)

    #point to label
    point = Image.open(os.path.join(point_root,point_list[j])).convert('L')
    point = np.array(point) #[250,250]
    point = np.where(point>0)
    point_label = np.ones(len(point[0]),dtype=int)
    point_x = np.array(point[0])
    point_y = np.array(point[1])
    point_xy = []
    for i in range(len(point_x)):
        point_xy.append([point_y[i],point_x[i]])
    point_xy = np.array(point_xy) #[18,2]

    #print("shape pooing", point_xy.shape)


########################################### mask bg seperatioin
    for k in range(3): #[1,250,250]
        #print(mask_init.shape)
        channel = mask_init[k:k+1,:,:].transpose(1, 2, 0)#    [250 250 1]
        maskBG = np.argwhere(channel == 0)#60338,3
        maskBG_x = np.array(maskBG[:,0])#60338
        maskBG_y = np.array(maskBG[:,1])
        maskBG_xy = []
        for i in range(len(maskBG_x)):
            maskBG_xy.append([maskBG_y[i],maskBG_x[i]])
        maskBG_xy=np.array(maskBG_xy) # [60338,2] 
        #print(maskBG_xy.shape) 

        maskBG_xy=random.choices(maskBG_xy,k=10)
        maskBG_label = np.zeros(10,dtype=int)
        #print(maskBG_xy)
        #3개의 마스크 중 하나의 마스크에서 백그라운드 랜덤 10개 뽑음
        cat_labels = np.concatenate((point_label, maskBG_label)) 
        cat_coords = np.concatenate((point_xy, maskBG_xy)) 
        #print(cat_coords)
        #print(cat_labels)

        predictor.set_image(image)
        masks, scores, logits = predictor.predict(point_labels= cat_labels,point_coords=cat_coords, multimask_output=True,)

        #print(masks.shape)
        for i, (mask, score) in enumerate(zip(masks, scores)):    
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            print(mask.shape)

            show_mask(mask, plt.gca())
            show_points(cat_coords, cat_labels, plt.gca())
            plt.axis('off')
            plt.show()
#############################################################
"""