import os
import numpy as np
import random
import torch

from scipy.ndimage.morphology import binary_fill_holes
import skimage.morphology as ski_morph
from skimage import measure
from skimage.measure import label
from scipy.spatial import Voronoi
from skimage import draw



def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


# borrowed from https://gist.github.com/pv/8036995
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def split_forward(model, input, size, overlap, outchannel=2):
    '''
    split the input image for forward passes
    '''

    b, c, h0, w0 = input.size()

    # zero pad for border patches
    pad_h = 0
    if h0 - size > 0 and (h0 - size) % (size - overlap) > 0:
        pad_h = (size - overlap) - (h0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, pad_h, w0))
        input = torch.cat((input, tmp), dim=2)

    if w0 - size > 0 and (w0 - size) % (size - overlap) > 0:
        pad_w = (size - overlap) - (w0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        input = torch.cat((input, tmp), dim=3)

    _, c, h, w = input.size()

    output = torch.zeros((input.size(0), outchannel, h, w))
    for i in range(0, h-overlap, size-overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h
        for j in range(0, w-overlap, size-overlap):
            c_end = j+size if j+size < w else w

            input_patch = input[:,:,i:r_end,j:c_end]
            input_var = input_patch.cuda()
            with torch.no_grad():
                output_patch = model(input_var)

            ind2_s = j+overlap//2 if j>0 else 0
            ind2_e = j+size-overlap//2 if j+size<w else w
            output[:,:,ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:,:,ind1_s-i:ind1_e-i, ind2_s-j:ind2_e-j]

    output = output[:,:,:h0,:w0].cuda()

    return output


def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random() for i in range(3)]
    return r, g, b


def show_figures(imgs, new_flag=False):
    import matplotlib.pyplot as plt
    if new_flag:
        for i in range(len(imgs)):
            plt.figure()
            plt.imshow(imgs[i])
    else:
        for i in range(len(imgs)):
            plt.figure(i+1)
            plt.imshow(imgs[i])

    plt.show()


# revised on https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self, shape=1):
        self.shape = shape
        self.reset()

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = 0

    def update(self, val, n=1):
        val = np.array(val)
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def write_txt(results, filename, mode='w'):
    """ Save the result of losses and F1 scores for each epoch/iteration
        results: a list of numbers
    """
    with open(filename, mode) as file:
        num = len(results)
        for i in range(num-1):
            file.write('{:.4f}\t'.format(results[i]))
        file.write('{:.4f}\n'.format(results[num-1]))


def save_results(header, all_result, test_results, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(header)
    with open(filename, mode) as file:
        # header
        file.write('Metrics:\t')
        for i in range(N - 1):
            file.write('{:s}\t'.format(header[i]))
        file.write('{:s}\n'.format(header[N - 1]))

        # average results
        file.write('Average results:\n')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(all_result[i]))
        file.write('{:.4f}\n'.format(all_result[N - 1]))
        file.write('\n')

        # results for each image
        for key, vals in sorted(test_results.items()):
            file.write('{:s}:\n'.format(key))
            for value in vals:
                file.write('\t{:.4f}'.format(value))
            file.write('\n')

def load_checkpoint(model, model_path):
    if not os.path.isfile(model_path):
        raise ValueError('Invalid checkpoint file: {}'.format(model_path))

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    try:
        epoch = checkpoint['epoch']
        tmp_state_dict = checkpoint['state_dict']
        print('loaded {}, epoch {}'.format(model_path, epoch))
    except:
        # The most naive way for serialization (especially for efficientdet)
        tmp_state_dict = checkpoint

    # create state_dict
    state_dict = {}

    # convert data_parallal to model
    for k in tmp_state_dict:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = tmp_state_dict[k]
        else:
            state_dict[k] = tmp_state_dict[k]

    model_state_dict = model.state_dict()
    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                try:
                    tmp = torch.zeros(model_state_dict[k].shape)  # create tensor with zero filled
                    tmp[:state_dict[k].shape[0]] = state_dict[k]  # fill valid
                    state_dict[k] = tmp
                    print('Load parameter partially {}, required shape {}, loaded shape {}'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                except:
                    print('Remain parameter (as random) {}'.format(k))  # when loaded state_dict has larger tensor
                    state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}'.format(k))

    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}'.format(k))
            state_dict[k] = model_state_dict[k]

    # load state_dict
    model.load_state_dict(state_dict, strict=False)

    return model

def save_checkpoint(save_path, model, epoch):
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    torch.save({'epoch': epoch + 1, 'state_dict': state_dict}, save_path)

def accuracy_object_level(pred, gt, hausdorff_flag=True):
    """ Compute the object-level metrics between predicted and
    groundtruth: dice, iou, hausdorff """
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    # get connected components

    pred_labeled = label(pred, connectivity=2)
    Ns = len(np.unique(pred_labeled)) - 1
    gt_labeled = label(gt, connectivity=2)
    Ng = len(np.unique(gt_labeled)) - 1

    # --- compute dice, iou, hausdorff --- #
    pred_objs_area = np.sum(pred_labeled>0)  # total area of objects in image
    gt_objs_area = np.sum(gt_labeled>0)  # total area of objects in groundtruth gt
    # print(pred_objs_area)
    # print(gt_objs_area)
    # compute how well groundtruth object overlaps its segmented object
    dice_g = 0.0
    iou_g = 0.0
    for i in range(1, Ng + 1):
        gt_i = np.where(gt_labeled == i, 1, 0)

        # cv2.imwrite('img.png', gt_i*255)
        overlap_parts = gt_i * pred_labeled

        # get intersection objects numbers in image
        obj_no = np.unique(overlap_parts)
        # print('1 : ', obj_no)
        obj_no = obj_no[obj_no != 0]
        # print('2 : ', obj_no)

        gamma_i = float(np.sum(gt_i)) / gt_objs_area
        # print('gamma_i : ', gamma_i)

        # show_figures((pred_labeled, gt_i, overlap_parts))

        if obj_no.size == 0:   # no intersection object
            dice_i = 0
            iou_i = 0

        else:
            # find max overlap object
            obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
            # print('obj_areas : ', obj_areas)

            seg_obj = obj_no[np.argmax(obj_areas)]  # segmented object number
            # print('seg_obj : ', seg_obj)

            pred_i = np.where(pred_labeled == seg_obj, 1, 0)  # segmented object

            overlap_area = np.max(obj_areas)  # overlap area

            dice_i = 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
            # print('dice_i : ', dice_i)
            iou_i = float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)
            # print('iou_i : ', iou_i)

        dice_g += gamma_i * dice_i
        iou_g += gamma_i * iou_i
        # print('dice_g : ', dice_g)
        # print('iou_g : ', iou_g)

    # compute how well segmented object overlaps its groundtruth object
    dice_s = 0.0
    iou_s = 0.0
    hausdorff_s = 0.0
    for j in range(1, Ns + 1):
        pred_j = np.where(pred_labeled == j, 1, 0)
        overlap_parts = pred_j * gt_labeled

        # get intersection objects number in gt
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        # show_figures((pred_j, gt_labeled, overlap_parts))

        sigma_j = float(np.sum(pred_j)) / pred_objs_area
        # no intersection object
        if obj_no.size == 0:
            dice_j = 0
            iou_j = 0

        else:
            # find max overlap gt
            gt_areas = [np.sum(overlap_parts == k) for k in obj_no]
            gt_obj = obj_no[np.argmax(gt_areas)]  # groundtruth object number
            gt_j = np.where(gt_labeled == gt_obj, 1, 0)  # groundtruth object

            overlap_area = np.max(gt_areas)  # overlap area

            dice_j = 2 * float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j))
            iou_j = float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j) - overlap_area)

        dice_s += sigma_j * dice_j
        iou_s += sigma_j * iou_j

    return (dice_g + dice_s) / 2, (iou_g + iou_s) / 2

def AJI_fast(gt, pred_arr):
    gt = label(gt)
    pred_arr = label(pred_arr)


    gs, g_areas = np.unique(gt, return_counts=True)  # gs is the instance number of gt, g_areas is the corresponding area.
    assert np.all(gs == np.arange(len(gs)))
    ss, s_areas = np.unique(pred_arr, return_counts=True)
    assert np.all(ss == np.arange(len(ss)))
    i_idx, i_cnt = np.unique(np.concatenate([gt.reshape(1, -1), pred_arr.reshape(1, -1)]),
                             return_counts=True, axis=1)
    i_arr = np.zeros(shape=(len(gs), len(ss)), dtype=np.int)

    i_arr[i_idx[0], i_idx[1]] += i_cnt  #
    u_arr = g_areas.reshape(-1, 1) + s_areas.reshape(1, -1) - i_arr
    iou_arr = 1.0 * i_arr / u_arr

    i_arr = i_arr[1:, 1:]  # remove background class intersection with other foreground class
    u_arr = u_arr[1:, 1:]  # remove background class union with other foreground class
    iou_arr = iou_arr[1:, 1:]
    # ipdb.set_trace()
    # if len(iou_arr) == 0:
    #     return 0
    try:
        j = np.argmax(iou_arr, axis=1)  # get the instance number in seg who has maximum iou with gt
    except ValueError:
        return 0
    c = np.sum(i_arr[np.arange(len(gs) - 1), j])
    u = np.sum(u_arr[np.arange(len(gs) - 1), j])
    used = np.zeros(shape=(len(ss) - 1), dtype=np.int)
    used[j] = 1
    u += (np.sum(s_areas[1:] * (1 - used)))
    return 1.0 * c / u



def compute_accuracy_1ch(pred, gt, radius=11, return_distance=False):
    """ compute detection accuracy: recall, precision, F1 """
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    # get connected components
    pred_labeled = ski_morph.label(pred)
    pred_regions = measure.regionprops(pred_labeled)
    # import matplotlib.pyplot as plt
    # plt.imshow(gt)
    # plt.show()
    # plt.imshow(pred_labeled)
    # plt.show()
    pred_points = []
    for region in pred_regions:
        pred_points.append(region.centroid)
    pred_points = np.array(pred_points)
    Np = pred_points.shape[0]

    gt_points = np.argwhere(gt == 1)
    # gt_labeled = ski_morph.label(gt)
    # gt_regions = measure.regionprops(gt_labeled)
    # gt_points = []
    # for region in gt_regions:
    #     gt_points.append(region.centroid)
    # gt_points = np.array(gt_points)

    Ng = gt_points.shape[0]
    TP = 0.0
    FN = 0.0
    d_list = []   # the distances between true locations and TP detections
    for i in range(Ng):   # for each gt point, find the nearest pred point
        if np.size(pred_points) == 0:
            FN += 1
            continue
        gt_point = gt_points[i, :]
        dist = np.linalg.norm(pred_points - gt_point, axis=1)
        if np.min(dist) < radius:  # the nearest pred point is in the radius of the gt point
            pred_idx = np.argmin(dist)
            pred_points = np.delete(pred_points, pred_idx, axis=0)   # delete the TP
            TP += 1
            d_list.append(np.min(dist))
        else:  # the nearest pred point is not in the radius
            FN += 1

    FP = Np - TP

    if return_distance:
        return TP, FP, FN, d_list
    else:
        return TP, FP, FN

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    ex = np.exp(x - np.max(x, axis=0))
    return ex / np.sum(ex, axis=0)

def cca(pred, fg_prob=False):
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)

    cutoff = 0.5
    min_area = 20
    if not fg_prob:
        pred = softmax(pred)
        pred = pred[1, :, :]

        pred[pred <= cutoff] = 0
        pred[pred > cutoff] = 1
    pred = pred.astype(int)

    pred_labeled = measure.label(pred)
    pred_labeled = ski_morph.remove_small_objects(pred_labeled, min_area)
    pred_labeled = binary_fill_holes(pred_labeled > 0)
    pred_labeled = measure.label(pred_labeled)
    return pred_labeled

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def mk_colored(instance_img):
    instance_img = instance_img.astype(np.int32)
    H, W = instance_img.shape[0], instance_img.shape[1]
    pred_colored_instance = np.zeros((H, W, 3))

    nuc_index = list(np.unique(instance_img))
    nuc_index.pop(0)

    for k in nuc_index:
        pred_colored_instance[instance_img == k, :] = np.array(get_random_color())

    return pred_colored_instance
