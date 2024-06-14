import warnings
from typing import Tuple, Literal

import cv2
import numpy as np
from scipy import ndimage
# from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation
import skimage.morphology as ski_morph
from skimage import measure
from skimage.measure import label
from skimage.segmentation import watershed
import torch

# from .tools import get_bounding_box, remove_small_objects

def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided.

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel.

    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out

def make_instance_hv(pred, hv_pred, object_size = 10, ksize = 21):
    """Process Nuclei Prediction with XY Coordinate Map and generate instance map (each instance has unique integer)

    Separate Instances (also overlapping ones) from binary nuclei map and hv map by using morphological operations and watershed

    Args:
        pred (np.ndarray): Prediction output, assuming. Shape: (H, W, 3)
            * channel 0 contain probability map of nuclei
            * channel 1 containing the regressed X-map
            * channel 2 containing the regressed Y-map
        object_size (int, optional): Smallest oject size for filtering. Defaults to 10
        k_size (int, optional): Sobel Kernel size. Defaults to 21
    Returns:
        np.ndarray: Instance map for one image. Each nuclei has own integer. Shape: (H, W)
    """
    # pred = np.array(pred, dtype=np.float32)
    #
    # blb_raw = pred[..., 0]
    # h_dir_raw = pred[..., 1]
    # v_dir_raw = pred[..., 2]
    #
    # # processing

    cutoff = 0.7
    min_area = 20

    pred = np.array(pred >= cutoff, dtype=np.int32)
    # blb = measurements.label(blb)[0]  # ndimage.label(blb)[0]
    # blb = remove_small_objects(blb, min_size=10)  # 10

    pred_labeled = measure.label(pred)
    pred_labeled = ski_morph.remove_small_objects(pred_labeled, min_area)
    pred_labeled = binary_fill_holes(pred_labeled > 0)
    pred_labeled = measure.label(pred_labeled)
    blb = pred_labeled.copy()

    blb[blb > 0] = 1  # background is 0 already
     #clone()? copy()

    h_dir_raw, v_dir_raw = hv_pred[0], hv_pred[1]

    h_dir = cv2.normalize(
        h_dir_raw,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    v_dir = cv2.normalize(
        v_dir_raw,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )

    # ksize = int((20 * scale_factor) + 1) # 21 vs 41
    # obj_size = math.ceil(10 * (scale_factor**2)) #10 vs 40

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

    sobelh = 1 - (
        cv2.normalize(
            sobelh,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.5, dtype=np.int32) #0.4

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measure.label(marker)
    marker = remove_small_objects(marker, min_size=object_size)

    proced_pred = watershed(dist, markers=marker, mask=blb)

    return pred_labeled, proced_pred, marker

def make_instance_sonnet(pred, pred_ord):
    """
    Process Nuclei Prediction with The ordinal map

    Args:
        pred: prediction output (NP branch)
        pred_ord: ordinal prediction output (ordinal branch)
    """

    cutoff = 0.5
    min_area = 20

    pred = np.array(pred >= cutoff, dtype=np.int32)
    # blb = measurements.label(blb)[0]  # ndimage.label(blb)[0]
    # blb = remove_small_objects(blb, min_size=10)  # 10

    pred_labeled = measure.label(pred)
    pred_labeled = ski_morph.remove_small_objects(pred_labeled, min_area)
    pred_labeled = binary_fill_holes(pred_labeled > 0)
    pred_labeled = measure.label(pred_labeled)
    blb = pred_labeled.copy()

    blb[blb > 0] = 1



    pred_ord = np.squeeze(pred_ord)
    distance = -pred_ord
    marker = np.copy(pred_ord)
    marker[marker <= 4] = 0 # ori: 4
    marker[marker > 4] = 1
    marker = binary_dilation(marker, iterations=1)
    # marker = binary_erosion(marker)
    # marker = binary_erosion(marker)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    # marker = cv2.morphologyEx(np.float32(marker), cv2.MORPH_OPEN, kernel)
    marker = measure.label(marker)
    marker = remove_small_objects(marker, min_size=10)


    blb = measure.label(blb)
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already

    markers = marker * blb

    proced_pred = watershed(distance, markers, mask=blb)

    return pred_labeled, proced_pred, marker

def make_instance_marker(pred, marker, th):
    """
    Process Nuclei Prediction with The ordinal map

    Args:
        pred: prediction output (NP branch)
        pred_ord: ordinal prediction output (ordinal branch)
    """

    cutoff = 0.5
    min_area = 20

    pred = np.array(pred >= cutoff, dtype=np.int32)
    # blb = measurements.label(blb)[0]  # ndimage.label(blb)[0]
    # blb = remove_small_objects(blb, min_size=10)  # 10

    pred_labeled = measure.label(pred)
    pred_labeled = ski_morph.remove_small_objects(pred_labeled, min_area)
    pred_labeled = binary_fill_holes(pred_labeled > 0)
    pred_labeled = measure.label(pred_labeled)
    blb = pred_labeled.copy()

    blb[blb > 0] = 1

    distance = -marker
    marker[marker <= th] = 0 # ori: 4
    marker[marker > th] = 1
    marker = binary_dilation(marker, iterations=1)
    # marker = binary_erosion(marker)
    # marker = binary_erosion(marker)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    # marker = cv2.morphologyEx(np.float32(marker), cv2.MORPH_OPEN, kernel)
    marker = measure.label(marker)
    marker = remove_small_objects(marker, min_size=10)


    blb = measure.label(blb)
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already

    markers = marker * blb

    proced_pred = watershed(distance, markers, mask=blb)

    return pred_labeled, proced_pred, marker