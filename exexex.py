# import os
# from PIL import Image
# import numpy as np
# import imageio
#
# # path = '/media/NAS/nas_70/open_datasetset/CPM/CPM 17/labels_instance'
# # img_list = os.listdir(path)
# #
# # save_path = '/media/NAS/nas_187/siwoo/train'
# # for img_name in img_list:
# #     img = Image.open(path + '/' + img_name)
# #
# #     img = np.array(img)
# #
# #     img1 = img[:300, :300]
# #     img1 = Image.fromarray(img1)
# #     img1.save(save_path+'/'+img_name[:-10]+'_0_label.png')
# #
# #     img2 = img[:300, -300:]
# #     img2 = Image.fromarray(img2)
# #     img2.save(save_path+'/'+img_name[:-10]+'_1_label.png')
# #
# #     img3 = img[-300:, :300]
# #     img3 = Image.fromarray(img3)
# #     img3.save(save_path+'/'+img_name[:-10]+'_2_label.png')
# #
# #     img4 = img[-300:, -300:]
# #     img4 = Image.fromarray(img4)
# #     img4.save(save_path+'/'+img_name[:-10]+'_3_label.png')
#
# dataset = 'CPM 17'
# data_dir ='/media/NAS/nas_70/open_datasetset/CPM/CPM 17'
# img_dir = f'/media/NAS/nas_70/open_dataset/CPM/{dataset}/images'
#
# label_instance_dir = '/media/NAS/nas_70/open_dataset/CPM/{:s}/labels_instance'.format(dataset)
# label_point_dir = '/media/NAS/nas_70/open_dataset/CPM/{:s}/labels_point_val'.format(dataset)
# label_binary_mask_dir = '/media/NAS/nas_70/open_dataset/CPM/{:s}/labels_binary_val'.format(dataset)
# label_vor_dir = '/media/NAS/nas70/open_dataset/CPM/{:s}/labels_voronoi'.format(dataset)
# label_geo_vor_dir = '/media/NAS/nas70/open_dataset/CPM/{:s}/labels_geo_voronoi_val'.format(dataset)
# label_cluster_dir = '/media/NAS/nas70/open_dataset/CPM/{:s}/labels_cluster'.format(dataset)
# label_geo_cluster_dir = '/media/NAS/nas_70/open_dataset/CPM/{:s}/labels_geo_cluster_val'.format(dataset)
# label_prob_dir = '/media/NAS/nas70/open_dataset/CPM/{:s}/labels_prob'.format(dataset)
#
# patch_folder = '/media/NAS/nas_70/open_dataset/CPM/{:s}/patches'.format(dataset)
# train_data_dir = '/media/NAS/nas_70/open_dataset/CPM/{:s}/via instance learning data_for_train/{:s}'.format(dataset,
#                                                                                                                dataset)
# split = '{:s}/train_val_test.json'.format(data_dir)
# stats_path = '{:s}/stats_val.csv'.format(data_dir)
#
#
# def split_patches(data_dir, save_dir, patch_size=250, h_num=4, w_num=4, post_fix="", ext="png"):
#     import math
#     """ split large image into small patches """
#
#     print("Spliting large {:s} images into small patches...".format(post_fix))
#
#     image_list = os.listdir(data_dir)
#     for image_name in image_list:
#         if image_name.startswith("."):
#             continue
#         name = image_name.split('.')[0]
#         if post_fix and name[-len(post_fix):] != post_fix:
#             continue
#         image_path = os.path.join(data_dir, image_name)
#         image = imageio.imread(image_path)
#         seg_imgs = []
#
#         # split into 16 patches of size 250x250
#         h, w = image.shape[0], image.shape[1]
#         h_overlap = math.ceil((h_num * patch_size - h) / (h_num-1))
#         w_overlap = math.ceil((w_num * patch_size - w) / (w_num-1))
#         for i in range(0, h - patch_size + 1, patch_size - h_overlap):
#             for j in range(0, w - patch_size + 1, patch_size - w_overlap):
#                 if len(image.shape) == 3:
#                     patch = image[i:i + patch_size, j:j + patch_size, :]
#                 else:
#                     patch = image[i:i + patch_size, j:j + patch_size]
#                 seg_imgs.append(patch)
#
#         for k in range(len(seg_imgs)):
#             if post_fix:
#                 imageio.imwrite(
#                     '{:s}/{:s}_{:d}_{:s}.{:s}'.format(save_dir, name[:-len(post_fix) - 1], k, post_fix, ext),
#                     seg_imgs[k])
#             else:
#                 imageio.imwrite('{:s}/{:s}_{:d}.{:s}'.format(save_dir, name, k, ext), seg_imgs[k])
#
# # split_patches(img_dir, '{:s}/img'.format('/media/NAS/nas_187/siwoo/train'), patch_size=300, h_num=2, w_num=2, post_fix='')
# split_patches(label_instance_dir, '{:s}/labels_instance'.format('/media/NAS/nas_187/siwoo/train'), patch_size=300, h_num=2, w_num=2, post_fix='label')
#
#
#
#
#
#

import numpy as np
import os

csv = [
'1__20231109_133121',
'1__20231108_231011',
'1__20231109_112239',
'1__20231108_213546',
'1__20231108_213828',
'1__20231108_213953',
'1__20231108_214150',
'1__20231109_111937',
'1__20231109_111658',
'1__20231108_214529',
'1__20231109_111430',
'1__20231108_214329',
'1__20231108_230807',
'1__20231109_111039',
'1__20231108_144804',
'1__20231108_144519',
'1__20231108_214905',
'1__20231108_144209',
'1__20231108_214711',
'1__20231108_135549',
'1__20231108_135358',
'1__20231108_215123',
'1__20231108_135101',
'1__20231108_225751',
'1__20231108_134742',
'1__20231108_215819',
'1__20231109_110832',
'1__20231108_134532',
'1__20231108_215415',
'1__20231108_134320',
'1__20231108_134140',
'1__20231108_133956',
'1__20231108_133757',
'1__20231108_133551',
'1__20231108_230558',
'1__20231108_132534',
'1__20231108_132327',
'1__20231108_132208',
'1__20231108_131958',
'1__20231108_131827',
'1__20231108_131603',
'1__20231108_221236',
'1__20231108_131433',
'1__20231108_145610',
'1__20231108_221013',
'1__20231108_231444',
'1__20231108_140033',
'1__20231108_131245',
'1__20231108_131056',
'1__20231108_132335',
'1__20231108_221400',
'1__20231108_132052',
'1__20231108_131859',
'1__20231108_215616',
'1__20231108_131641',
'1__20231109_131336',
'1__20231108_220234',
'1__20231108_220026',
'1__20231108_131429',
'1__20231108_220714',
'1__20231108_220458',
'1__20231108_222632',
'1__20231108_222800',
'1__20231108_223117',
'1__20231108_222950',
'1__20231108_231305',
'1__20231108_223346',
'1__20231108_223509',
'1__20231109_004124',
'1__20231109_131037',
'1__20231108_221817',
'1__20231108_141158',
'1__20231109_003745',
'1__20231108_140954',
'1__20231108_140654',
'1__20231108_140455',
'1__20231108_142812',
'1__20231109_130813',
'1__20231108_142622',
'1__20231108_142425',
'1__20231108_142215',
'1__20231108_142106',
'1__20231108_201037',
'1__20231108_141910',
'1__20231108_200722',
'1__20231108_141702',
'1__20231108_200950',
'1__20231109_130647',
'1__20231108_201747',
'1__20231108_144019',
'1__20231108_143743',
'1__20231108_143511',
'1__20231108_202359',
'1__20231108_143159',
'1__20231108_202210',
'1__20231108_143006',
'1__20231108_145449',
'1__20231109_132907',
'1__20231109_130457',
'1__20231109_011251',
'1__20231108_145228',
'1__20231108_235442',
'1__20231108_144957',
'1__20231108_145836',
'1__20231108_235030',
'1__20231108_212912',
'1__20231109_130234',
'1__20231108_201546',
'1__20231108_233516',
'1__20231108_222009',
'1__20231108_221626',
'1__20231108_222443',
'1__20231108_222235',
'1__20231108_135806',
'1__20231108_235241',
'1__20231109_130018',
'1__20231108_135634',
'1__20231108_135403',
'1__20231108_135142',
'1__20231108_133902',
'1__20231108_134101',
'1__20231108_133705',
'1__20231109_125848',
'1__20231109_125722',
'1__20231109_125618',
'1__20231109_125434',
'1__20231109_133007',
'1__20231109_133140',
'1__20231109_132741',
'1__20231109_132649',
'1__20231109_132551',
'1__20231109_132419',
'1__20231109_132206',
'1__20231109_131951',
'1__20231109_131646',
'1__20231109_131237',
'1__20231109_131048',
'1__20231109_130932',
'1__20231109_130714',
'1__20231109_130537',
'1__20231109_132519',
'1__20231109_131427',
'1__20231109_130329',
'1__20231109_130200',
'1__20231109_130030',
'1__20231109_125744',
'1__20231109_125343',
'1__20231109_125217',
'1__20231109_141400',
'1__20231109_141104',
'1__20231109_140856',
'1__20231109_132237',
'1__20231108_231818',
'1__20231109_140655',
'1__20231109_140500',
'1__20231109_140237',
'1__20231109_140119',
'1__20231109_135844',
'1__20231109_135633',
'1__20231109_135410',
'1__20231109_135230',
'1__20231109_134940',
'1__20231109_132119',
'1__20231109_134717',
'1__20231109_134533',
'1__20231109_134346',
'1__20231108_231637',
'1__20231109_134140',
'1__20231109_134026',
'1__20231109_133806',
'1__20231108_232108',
'1__20231109_133605',
'1__20231109_133359',
'1__20231109_131946',
'1__20231109_111302',
'1__20231109_115116',
'1__20231108_225939',
'1__20231109_114900',
'1__20231109_114359',
'1__20231109_114640',
'1__20231109_114200',
'1__20231109_113931',
'1__20231108_231943',
'1__20231108_232418',
'1__20231109_131640',
'1__20231108_232239',
'1__20231109_113543',
'1__20231108_223654',
'1__20231108_223904',
'1__20231108_224139',
'1__20231109_113340',
'1__20231108_224335',
'1__20231108_224511',
'1__20231109_113143',
'1__20231109_112935',
'1__20231109_131542',
'1__20231108_224712',
'1__20231108_224907',
'1__20231108_230403',
'1__20231108_225137',
'1__20231108_225332',
'1__20231108_230149',
'1__20231109_112729',
'1__20231109_112556',
'1__20231109_112416',
'1__20231108_213255',
]

aaa = os.listdir('/media/NAS/nas_32/PATHOLOGY_DATA/AMC_Liver_Research_2024/Scanfile_tiff_23-001526')
iii = 0
for i in range(len(aaa)):
    print(aaa[i])
#     name = aaa[i].split(';')
#
#     if name[-1][:-5] in csv:
#         iii +=1
#     else:
#         # print(name[-1][:-5])
#         print(aaa[i])
# print(iii)