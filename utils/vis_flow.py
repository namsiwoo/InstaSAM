# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03
# Reference: https://github.com/tomrunia/OpticalFlow_Visualization

import numpy as np
import skimage

def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi

    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.shape[1]):

        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1

        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)

def find_centroids_with_refinement(displacement, iterations=300):
    # iteration: the number of refinement steps (u), set to any integer >= 100.

    height, width = displacement.shape[1:3]

    # 1. initialize centroids as their coordinates
    centroid_y = np.repeat(np.expand_dims(np.arange(height), 1), width, axis=1).astype(np.float32)
    centroid_x = np.repeat(np.expand_dims(np.arange(width), 0), height, axis=0).astype(np.float32)

    for i in range(iterations):

        # 2. find numbers after the decimals
        uy = np.ceil(centroid_y).astype(np.int32)
        dy = np.floor(centroid_y).astype(np.int32)
        y_c = centroid_y - dy

        ux = np.ceil(centroid_x).astype(np.int32)
        dx = np.floor(centroid_x).astype(np.int32)
        x_c = centroid_x - dx

        # 3. move centroids
        centroid_y += displacement[0][uy, ux] * y_c * x_c + \
                      displacement[0][dy, ux] *(1 - y_c) * x_c + \
                      displacement[0][uy, dx] * y_c * (1 - x_c) + \
                      displacement[0][dy, dx] * (1 - y_c) * (1 - x_c)

        centroid_x += displacement[1][uy, ux] * y_c * x_c + \
                      displacement[1][dy, ux] *(1 - y_c) * x_c + \
                      displacement[1][uy, dx] * y_c * (1 - x_c) + \
                      displacement[1][dy, dx] * (1 - y_c) * (1 - x_c)

        # 4. bound centroids
        centroid_y = np.clip(centroid_y, 0, height-1)
        centroid_x = np.clip(centroid_x, 0, width-1)

    centroid_y = np.round(centroid_y).astype(np.int32)
    centroid_x = np.round(centroid_x).astype(np.int32)

    return np.stack([centroid_y, centroid_x], axis=0)

def cluster_centroids(centroids, displacement, thres=2.5):
    # thres: threshold for grouping centroid (see supp)

    dp_strength = np.sqrt(displacement[1] ** 2 + displacement[0] ** 2)
    height, width = dp_strength.shape

    weak_dp_region = dp_strength < thres

    dp_label = skimage.measure.label(weak_dp_region, connectivity=1, background=0)
    dp_label_1d = dp_label.reshape(-1)

    centroids_1d = centroids[0]*width + centroids[1]

    clusters_1d = dp_label_1d[centroids_1d]

    cluster_map = compress_range(clusters_1d.reshape(height, width) + 1)

    return to_one_hot(cluster_map)

def compress_range(arr):
    uniques = np.unique(arr)
    maximum = np.max(uniques)

    d = np.zeros(maximum+1, np.int32)
    d[uniques] = np.arange(uniques.shape[0])

    out = d[arr]
    return out - np.min(out)

def to_one_hot(sparse_integers, maximum_val=None, dtype=np.bool):

    if maximum_val is None:
        maximum_val = np.max(sparse_integers) + 1

    src_shape = sparse_integers.shape

    flat_src = np.reshape(sparse_integers, [-1])
    src_size = flat_src.shape[0]

    one_hot = np.zeros((maximum_val, src_size), dtype)
    one_hot[flat_src, np.arange(src_size)] = 1

    one_hot = np.reshape(one_hot, [maximum_val] + list(src_shape))

    return one_hot