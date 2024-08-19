import os, random
import time
import shutil

import torch
import numpy as np
from torch.optim import SGD, Adam, AdamW
# from tensorboardX import SummaryWriter
from scipy.ndimage.morphology import binary_fill_holes
import skimage.morphology as ski_morph
from skimage import measure
from skimage.measure import label
from scipy.optimize import linear_sum_assignment
import sod_metric
from numba import jit
class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    # if flatten:
    #     ret = ret.view(-1, ret.shape[-1])

    return ret



def calc_cod(y_pred, y_true):
    batchsize = y_true.shape[0]

    metric_FM = sod_metric.Fmeasure()
    metric_WFM = sod_metric.WeightedFmeasure()
    metric_SM = sod_metric.Smeasure()
    metric_EM = sod_metric.Emeasure()
    metric_MAE = sod_metric.MAE()
    with torch.no_grad():
        assert y_pred.shape == y_true.shape

        for i in range(batchsize):
            true, pred = \
                y_true[i, 0].cpu().data.numpy() * 255, y_pred[i, 0].cpu().data.numpy() * 255

            metric_FM.step(pred=pred, gt=true)
            metric_WFM.step(pred=pred, gt=true)
            metric_SM.step(pred=pred, gt=true)
            metric_EM.step(pred=pred, gt=true)
            metric_MAE.step(pred=pred, gt=true)

        fm = metric_FM.get_results()["fm"]
        wfm = metric_WFM.get_results()["wfm"]
        sm = metric_SM.get_results()["sm"]
        em = metric_EM.get_results()["em"]["curve"].mean()
        mae = metric_MAE.get_results()["mae"]

    return sm, em, wfm, mae


from sklearn.metrics import precision_recall_curve


def calc_f1(y_pred,y_true):
    batchsize = y_true.shape[0]
    with torch.no_grad():
        assert y_pred.shape == y_true.shape
        f1, auc = 0, 0
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        for i in range(batchsize):
            true = y_true[i].flatten()
            true = true.astype(np.int)
            pred = y_pred[i].flatten()

            precision, recall, thresholds = precision_recall_curve(true, pred)

            # auc
            auc += roc_auc_score(true, pred)
            # auc += roc_auc_score(np.array(true>0).astype(np.int), pred)
            f1 += max([(2 * p * r) / (p + r+1e-10) for p, r in zip(precision, recall)])

    return f1/batchsize, auc/batchsize, np.array(0), np.array(0)

def calc_fmeasure(y_pred,y_true):
    batchsize = y_true.shape[0]

    mae, preds, gts = [], [], []
    with torch.no_grad():
        for i in range(batchsize):
            gt_float, pred_float = \
                y_true[i, 0].cpu().data.numpy(), y_pred[i, 0].cpu().data.numpy()

            # # MAE
            mae.append(np.sum(cv2.absdiff(gt_float.astype(float), pred_float.astype(float))) / (
                        pred_float.shape[1] * pred_float.shape[0]))
            # mae.append(np.mean(np.abs(pred_float - gt_float)))
            #
            pred = np.uint8(pred_float * 255)
            gt = np.uint8(gt_float * 255)

            pred_float_ = np.where(pred > min(1.5 * np.mean(pred), 255), np.ones_like(pred_float),
                                   np.zeros_like(pred_float))
            gt_float_ = np.where(gt > min(1.5 * np.mean(gt), 255), np.ones_like(pred_float),
                                 np.zeros_like(pred_float))

            preds.extend(pred_float_.ravel())
            gts.extend(gt_float_.ravel())

        RECALL = recall_score(gts, preds)
        PERC = precision_score(gts, preds)

        fmeasure = (1 + 0.3) * PERC * RECALL / (0.3 * PERC + RECALL)
        MAE = np.mean(mae)

    return fmeasure, MAE, np.array(0), np.array(0)

from sklearn.metrics import roc_auc_score,recall_score,precision_score
import cv2
def calc_ber(y_pred, y_true):
    batchsize = y_true.shape[0]
    y_pred, y_true = y_pred.permute(0, 2, 3, 1).squeeze(-1), y_true.permute(0, 2, 3, 1).squeeze(-1)
    with torch.no_grad():
        assert y_pred.shape == y_true.shape
        pos_err, neg_err, ber = 0, 0, 0
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        for i in range(batchsize):
            true = y_true[i].flatten()
            pred = y_pred[i].flatten()

            TP, TN, FP, FN, BER, ACC = get_binary_classification_metrics(pred * 255,
                                                                         true * 255, 125)
            pos_err += (1 - TP / (TP + FN)) * 100
            neg_err += (1 - TN / (TN + FP)) * 100

    return pos_err / batchsize, neg_err / batchsize, (pos_err + neg_err) / 2 / batchsize, np.array(0)

def get_binary_classification_metrics(pred, gt, threshold=None):
    if threshold is not None:
        gt = (gt > threshold)
        pred = (pred > threshold)
    TP = np.logical_and(gt, pred).sum()
    TN = np.logical_and(np.logical_not(gt), np.logical_not(pred)).sum()
    FN = np.logical_and(gt, np.logical_not(pred)).sum()
    FP = np.logical_and(np.logical_not(gt), pred).sum()
    BER = cal_ber(TN, TP, FN, FP)
    ACC = cal_acc(TN, TP, FN, FP)
    return TP, TN, FP, FN, BER, ACC

def cal_ber(tn, tp, fn, fp):
    return  0.5*(fp/(tn+fp) + fn/(fn+tp))

def cal_acc(tn, tp, fn, fp):
    return (tp + tn) / (tp + tn + fp + fn)

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _eval_pr(y_pred, y, num):
    prec, recall = torch.zeros(num), torch.zeros(num)
    thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() +
                                                                1e-20)
    return prec, recall

def _S_object(pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1 - gt)
    u = gt.mean()
    Q = u * o_fg + (1 - u) * o_bg
    return Q

def _object(pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

    return score

def _S_region(pred, gt):
    X, Y = _centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
    p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
    Q1 = _ssim(p1, gt1)
    Q2 = _ssim(p2, gt2)
    Q3 = _ssim(p3, gt3)
    Q4 = _ssim(p4, gt4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q

def _centroid(gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        X = torch.eye(1) * round(cols / 2)
        Y = torch.eye(1) * round(rows / 2)
    else:
        total = gt.sum()
        i = torch.from_numpy(np.arange(0, cols)).float().cuda()
        j = torch.from_numpy(np.arange(0, rows)).float().cuda()
        X = torch.round((gt.sum(dim=0) * i).sum() / total + 1e-20)
        Y = torch.round((gt.sum(dim=1) * j).sum() / total + 1e-20)
    return X.long(), Y.long()


def _divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    area = h * w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4


def _dividePrediction(pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB


def _ssim(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q

def _eval_e(y_pred, y, num):
    score = torch.zeros(num)
    thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_pred_th = (y_pred >= thlist[i]).float()
        fm = y_pred_th - y_pred_th.mean()
        gt = y - y.mean()
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
    return score

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
    # pred_labeled[pred_labeled>0] = 1
    Ns = len(np.unique(pred_labeled)) - 1
    gt_labeled = label(gt, connectivity=2)
    # gt_labeled[gt_labeled>0] = 1
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

def AJI_fast(gt, pred_arr, name):
    gt = label(gt)
    pred_arr = label(pred_arr)

    # gt[gt>0] = 1
    # pred_arr[pred_arr>0] = 1

    gs, g_areas = np.unique(gt, return_counts=True)  # gs is the instance number of gt, g_areas is the corresponding area.
    # assert np.all(gs == np.arange(len(gs))), name
    if np.all(gs == np.arange(len(gs))):
        ss, s_areas = np.unique(pred_arr, return_counts=True)
        # assert np.all(ss == np.arange(len(ss))), name
        if np.all(ss == np.arange(len(ss))):
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
        else:
            print(name, gs)
            return 0.0
    else:
        print(name)
        return 0.0


def get_fast_pq(true, pred, match_iou=0.5):
    """
    `match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = label(true)
    pred = label(pred)


    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    # if there is no background, fixing by adding it
    if 0 not in pred_id_list:
        pred_id_list = [0] + pred_id_list

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)


    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    print(np.unique(t_mask), true_id_list)
    for true_id in true_id_list[1:]:  # 0-th is background
        try:
            t_mask = true_masks[true_id]
            pred_true_overlap = pred[t_mask > 0]
            pred_true_overlap_id = np.unique(pred_true_overlap)
            pred_true_overlap_id = list(pred_true_overlap_id)
            for pred_id in pred_true_overlap_id:
                if pred_id == 0:  # ignore
                    continue  # overlaping background
                p_mask = pred_masks[pred_id]
                total = (t_mask + p_mask).sum()
                inter = (t_mask * p_mask).sum()
                iou = inter / (total - inter)
                pairwise_iou[true_id - 1, pred_id - 1] = iou
        except:
            print(np.unique(t_mask), true_id_list)

    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1.0e-6)  # good practice?
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]

def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """
    Average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Args:
        masks_true (list of np.ndarrays (int) or np.ndarray (int)):
            where 0=NO masks; 1,2... are mask labels
        masks_pred (list of np.ndarrays (int) or np.ndarray (int)):
            np.ndarray (int) where 0=NO masks; 1,2... are mask labels

    Returns:
        ap (array [len(masks_true) x len(threshold)]):
            average precision at thresholds
        tp (array [len(masks_true) x len(threshold)]):
            number of true positives at thresholds
        fp (array [len(masks_true) x len(threshold)]):
            number of false positives at thresholds
        fn (array [len(masks_true) x len(threshold)]):
            number of false negatives at thresholds
    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]

    if len(masks_true) != len(masks_pred):
        raise ValueError(
            "metrics.average_precision requires len(masks_true)==len(masks_pred)")

    ap = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))

    for n in range(len(masks_true)):
        #_,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k, th in enumerate(threshold):
                tp[n, k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n]+ 1e-10)

    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return ap, tp, fp, fn

@jit(nopython=True)
def _label_overlap(x, y):
    """Fast function to get pixel overlaps between masks in x and y.

    Args:
        x (np.ndarray, int): Where 0=NO masks; 1,2... are mask labels.
        y (np.ndarray, int): Where 0=NO masks; 1,2... are mask labels.

    Returns:
        overlap (np.ndarray, int): Matrix of pixel overlaps of size [x.max()+1, y.max()+1].
    """
    # put label arrays into standard form then flatten them
    #     x = (utils.format_labels(x)).ravel()
    #     y = (utils.format_labels(y)).ravel()
    x = x.ravel()
    y = y.ravel()

    # preallocate a "contact map" matrix
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)

    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    """Calculate the intersection over union of all mask pairs.

    Parameters:
        masks_true (np.ndarray, int): Ground truth masks, where 0=NO masks; 1,2... are mask labels.
        masks_pred (np.ndarray, int): Predicted masks, where 0=NO masks; 1,2... are mask labels.

    Returns:
        iou (np.ndarray, float): Matrix of IOU pairs of size [x.max()+1, y.max()+1].

    How it works:
        The overlap matrix is a lookup table of the area of intersection
        between each set of labels (true and predicted). The true labels
        are taken to be along axis 0, and the predicted labels are taken
        to be along axis 1. The sum of the overlaps along axis 0 is thus
        an array giving the total overlap of the true labels with each of
        the predicted labels, and likewise the sum over axis 1 is the
        total overlap of the predicted labels with each of the true labels.
        Because the label 0 (background) is included, this sum is guaranteed
        to reconstruct the total area of each label. Adding this row and
        column vectors gives a 2D array with the areas of every label pair
        added together. This is equivalent to the union of the label areas
        except for the duplicated overlap area, so the overlap matrix is
        subtracted to find the union matrix.
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def _true_positive(iou, th):
    """Calculate the true positive at threshold th.

    Args:
        iou (float, np.ndarray): Array of IOU pairs.
        th (float): Threshold on IOU for positive label.

    Returns:
        tp (float): Number of true positives at threshold.

    How it works:
        (1) Find minimum number of masks.
        (2) Define cost matrix; for a given threshold, each element is negative
            the higher the IoU is (perfect IoU is 1, worst is 0). The second term
            gets more negative with higher IoU, but less negative with greater
            n_min (but that's a constant...).
        (3) Solve the linear sum assignment problem. The costs array defines the cost
            of matching a true label with a predicted label, so the problem is to
            find the set of pairings that minimizes this cost. The scipy.optimize
            function gives the ordered lists of corresponding true and predicted labels.
        (4) Extract the IoUs from these pairings and then threshold to get a boolean array
            whose sum is the number of true positives that is returned.
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp


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
    # nuc_index.pop(0)


    for k in nuc_index:
        pred_colored_instance[instance_img == k, :] = np.array(get_random_color())

    return pred_colored_instance

def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random() for i in range(3)]
    return r, g, b