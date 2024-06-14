# # # make csv files for GNAH, TCGA
# # import os, glob
# # import numpy as np
# # import csv
# #
# # path = '/media/NAS/nas_187/PATHOLOGY_DATA/ASAN Melanoma annotated/TCGA_GNAH_result/TCGA/TCGA_TIL_patchs_0830/TIL_patch'
# # path_list = glob.glob(os.path.join(path, '*.png'))
# # print(path_list)
# # path_list = np.array(path_list)
# # path_list = np.expand_dims(path_list, axis=1)
# #
# # f = open(os.path.join('/media/NAS/nas_187/PATHOLOGY_DATA/ASAN Melanoma annotated/TCGA_GNAH_result/TCGA/TCGA_TIL_patchs_0830', 'validation_til.csv'), 'w', newline='')
# # data = [['가나다라','12345'],['마바사아','678910']]
# # writer = csv.writer(f)
# # writer.writerows(path_list)
# # f.close()
# import glob
#
# import matplotlib.pyplot as plt
# # #move img in 0 or 1
# # import numpy as np
# # import os, csv, json
# # from PIL import Image
# #
# # patch_folder = '/media/NAS/nas_187/PATHOLOGY_DATA/ASAN Melanoma annotated/GNAH/GNAH_patch/GNAH patches'
# # points_folder = '/media/NAS/nas_187/PATHOLOGY_DATA/ASAN Melanoma annotated/GNAH/GNAH_patch/GNAH points'
# # save_path = '/media/NAS/nas_187/PATHOLOGY_DATA/ASAN Melanoma annotated/TCGA_GNAH_result/GNAH/GNAH_TIL_patchs_0830'
# #
# # img_list = os.listdir(patch_folder)
# #
# # for img_name in img_list:
# #     point_img = Image.open(os.path.join(points_folder, img_name))
# #     img = Image.open(os.path.join(patch_folder, img_name))
# #     if np.sum(np.array(point_img)) == 0:
# #         img.save(os.path.join(save_path, str(0)+'_img', img_name))
# #
# #     else:
# #         img.save(os.path.join(save_path, str(1)+'_img', img_name))
#
#
# import numpy as np
# import os, csv, json
#
# import torch
# from PIL import Image
#
#
# def normalizeStaining(img, saveFile=None, Io=240, alpha=1, beta=0.15):
#     ''' Normalize staining appearence of H&E stained images
#
#     Example use:
#         see test.py
#
#     Input:
#         I: RGB input image
#         Io: (optional) transmitted light intensity
#
#     Output:
#         Inorm: normalized image
#         H: hematoxylin image
#         E: eosin image
#
#     Reference:
#         A method for normalizing histology slides for quantitative analysis. M.
#         Macenko et al., ISBI 2009
#     '''
#
#     HERef = np.array([[0.5626, 0.2159],
#                       [0.7201, 0.8012],
#                       [0.4062, 0.5581]])
#
#     maxCRef = np.array([1.9705, 1.0308])
#
#     # define height and width of image
#     h, w, c = img.shape
#
#     # reshape image
#     img = img.reshape((-1, 3))
#
#     # calculate optical density
#     OD = -np.log((img.astype(np.float) + 1) / Io)
#
#     # remove transparent pixels
#     ODhat = OD[~np.any(OD < beta, axis=1)]
#
#     # compute eigenvectors
#     eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
#
#     # eigvecs *= -1
#
#     # project on the plane spanned by the eigenvectors corresponding to the two
#     # largest eigenvalues
#     That = ODhat.dot(eigvecs[:, 1:3])
#
#     phi = np.arctan2(That[:, 1], That[:, 0])
#
#     minPhi = np.percentile(phi, alpha)
#     maxPhi = np.percentile(phi, 100 - alpha)
#
#     vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
#     vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
#
#     # a heuristic to make the vector corresponding to hematoxylin first and the
#     # one corresponding to eosin second
#     if vMin[0] > vMax[0]:
#         HE = np.array((vMin[:, 0], vMax[:, 0])).T
#     else:
#         HE = np.array((vMax[:, 0], vMin[:, 0])).T
#
#     # rows correspond to channels (RGB), columns to OD values
#     Y = np.reshape(OD, (-1, 3)).T
#
#     # determine concentrations of the individual stains
#     C = np.linalg.lstsq(HE, Y, rcond=None)[0]
#
#     # normalize stain concentrations
#     maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
#     tmp = np.divide(maxC, maxCRef)
#     C2 = np.divide(C, tmp[:, np.newaxis])
#
#     # recreate the image using reference mixing matrix
#     Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
#     Inorm[Inorm > 255] = 254
#     Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)
#
#     # unmix hematoxylin and eosin
#     H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
#     H[H > 255] = 254
#     H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
#
#     E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
#     E[E > 255] = 254
#     E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
#
#     if saveFile is not None:
#         Image.fromarray(Inorm).save(saveFile + '.png')
#         Image.fromarray(H).save(saveFile + '_H.png')
#         Image.fromarray(E).save(saveFile + '_E.png')
#
#     return Inorm, H, E
#
# target_dir = 'no_TIL_patch'
# target_img_list = '0_img'
#
# img_path = '/media/NAS/nas_187/PATHOLOGY_DATA/ASAN Melanoma annotated/TCGA_GNAH_result/TCGA/TCGA_TIL_patchs_0830'
# TIL_patch_list = glob.glob(os.path.join(img_path, target_dir, '*.png'))
# img_0_list = os.listdir(os.path.join(img_path, target_img_list))
# print(img_0_list)
#
# for til_path in TIL_patch_list:
#     til_crop_patch_name = til_path.split('/')[-1]
#     patch_name = til_path.split('/')[-1][:-10]
#
#     til_crop_img_ori = np.array(Image.open(os.path.join(img_path, target_dir, til_crop_patch_name)))
#     coord = np.where(til_crop_img_ori != [255, 255, 255])
#     y, x, = coord[0], coord[1]
#     til_crop_img = til_crop_img_ori[np.min(y):np.max(y), np.min(x):np.max(x)]
#     h, w = til_crop_img.shape[0], til_crop_img.shape[1]
#
#     # print(patch_name)
#     if patch_name+'.png' in img_0_list:
#         print(patch_name)
#         patch_img = np.array(Image.open(os.path.join(img_path, target_img_list, patch_name+'.png')))
#
#         sam_loc = np.zeros_like(patch_img)
#
#         c_y, c_x = 0, 0
#         for i in range(patch_img.shape[0]-h):
#             for j in range(patch_img.shape[1]-w):
#                 # print(patch_img[i:i+h, j:j+w].shape, til_crop_img.shape)
#                 # print(np.array_equal(patch_img[i:i+h, j:j+w], til_crop_img))
#                 if np.sum(patch_img[i:i+h, j:j+w] == til_crop_img) > h*w*3-1:
#                     # print(patch_img[i:i+h, j:j+w] == til_crop_img)
#                     # plt.clf()
#                     # plt.subplot(1, 2, 1)
#                     # plt.imshow(patch_img[i:i+h, j:j+w])
#                     # plt.subplot(1, 2, 2)
#                     # plt.imshow(til_crop_img)
#                     # plt.show()
#
#                     sam_loc[i:i+h, j:j+w, :] = 255
#                     c_y = i
#                     c_x = j
#
#         norm_patch_img, _, _ = normalizeStaining(patch_img)
#
#         norm_til_crop_img = norm_patch_img[c_y:c_y+h, c_x:c_x+w]
#         til_crop_img_ori[np.min(y):np.max(y), np.min(x):np.max(x)] = norm_til_crop_img
#     til_crop_img_ori = Image.fromarray(til_crop_img_ori).convert('RGB')
#     til_crop_img_ori.save(os.path.join(img_path, target_dir, til_crop_patch_name))
#
#     # plt.clf()
#     # plt.subplot(1, 2, 1)
#     # plt.imshow(til_crop_img_ori)
#     # plt.subplot(1, 2, 2)
#     # plt.imshow(norm_til_crop_img)
#     # # plt.subplot(1, 3, 3)
#     # # plt.imshow(sam_loc)
#     # plt.show()
#
#
#
#
#

import numpy as np
import sklearn

cm = np.array([[14234, 52], [3510, 502]])
tn = cm[0, 0]
tp = cm[1, 1]
fp = cm[0, 1]
fn = cm[1, 0]

precision = tp/(tp+fp)
recall = tp/(tp+fn)
specificity = tn/(tn+fp)
f1_score = 2*(precision*recall)/(precision+recall)
accuracy = (tp+tn)/(tp+tn+fp+fn)

print(round(accuracy*100, 1), round(f1_score*100, 1), round(precision*100, 1), round(recall*100, 1), round(specificity*100, 1))


