import logging
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer, PromptEncoder
from utils.loss_offset import offset_Loss, offset_Loss_sonnet

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple


def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

def make_pseudo_offset(pred, vornoi):
    from skimage import measure
    from scipy.ndimage.morphology import binary_fill_holes
    import skimage.morphology as ski_morph
    cutoff = 0.5
    min_area = 20


    bg = pred < 0.5
    pred = pred > cutoff
    pred = measure.label(pred.detach().cpu().numpy())
    pred = ski_morph.remove_small_objects(pred, min_area)
    pred = binary_fill_holes(pred > 0)
    pred = torch.from_numpy(pred).to(vornoi.device.index)

    vornoi = vornoi
    vornoi[vornoi >0 ] = 1
    vornoi = vornoi.detach().cpu().numpy()

    region = np.zeros_like(vornoi)
    for b in range(vornoi.shape[0]):
        region[b] = measure.label(vornoi[b])

    # import matplotlib.pyplot as plt
    # plt.imshow(region[0])
    # plt.colorbar()
    # plt.savefig('/media/NAS/nas_187/siwoo/2023/result/transformer_freeze_new_h2_pseudo_MO/img/0/pseudo_offset.png')

    region = torch.from_numpy(region).to(pred.device.index)
    return pred.squeeze(1)*region, bg+pred

def make_point_prompt(points, only_fg=True):
    if only_fg == True:
        point_coords = []
        for index in (torch.unique(points[0])[1:]):
            point = (points[0] == index) * 1
            y, x = torch.nonzero(point, as_tuple=True)
            point_coords.append([x[0], y[0]])
        point_labels = np.ones(len(point_coords), dtype=int)

        point_coords = torch.from_numpy(np.array(point_coords, dtype=np.float32)).unsqueeze(0)
        point_coords = apply_coords_torch(point_coords,(points.shape[1], points.shape[2]), 1024)
        point_coords = point_coords.transpose(1, 0)

        point_labels = torch.from_numpy(point_labels).unsqueeze(1)

        return point_coords, point_labels

    else:
        point_coords = []
        for index in (torch.unique(points[0])[1:]):
            point = (points[0] == index) * 1
            y, x = torch.nonzero(point, as_tuple=True)
            point_coords.append([x[0], y[0]])

        point_coords = torch.from_numpy(np.array(point_coords, dtype=np.float32)).unsqueeze(0)
        point_coords = apply_coords_torch(point_coords, (points.shape[1], points.shape[2]), 1024)
        point_coords_2 = []


        #### less bg 4
        num_bg = min(len(point_coords)-1, 4)

        for i in range(point_coords.shape[1]):

        ### less bg 4
            pp = list(range(point_coords.shape[1]))
            pp.remove(i)
            pp = random.sample(pp, num_bg)
            pp.append(i)
            point_coords_2.append(point_coords[:, pp, :])
        point_coords_2 = torch.cat(point_coords_2, dim=0)

        point_labels = torch.zeros((len(point_coords_2), num_bg+1))
        point_labels[:, -1] = 1


        ### full bg
        #     point_coords_2.append(point_coords)
        # point_coords_2 = torch.cat(point_coords_2, dim=0)
        #
        # point_labels = torch.eye(len(point_coords_2))

        return point_coords_2, point_labels

def make_pseudo_gt(mask_prompt):
    # bg = torch.zeros(1, 224, 224).to(mask_prompt.device.index) + 0.3
    # bg = (torch.argmax(torch.cat([bg, mask_prompt], dim=0), dim=0) < 1) * 2
    #
    # ignored = torch.zeros(1, 224, 224).to(mask_prompt.device.index) + 0.7
    # ignored = (torch.argmax(torch.cat([ignored, mask_prompt.device.index], dim=0), dim=0) < 1) * 1
    #
    # pseudo_gt = torch.argmax(torch.cat([ignored, bg, mask_prompt.squeeze(1)], dim=0), dim=0) - 1

    mask_prompt = torch.sigmoid(mask_prompt.squeeze(1))
    bg = torch.zeros(1, mask_prompt.shape[-2], mask_prompt.shape[-1]).to(mask_prompt.device.index) + 0.5
    pseudo_gt = torch.argmax(torch.cat([bg, mask_prompt], dim=0), dim=0)

    fg = (mask_prompt > 0.5).float()
    overlap = torch.sum(fg, dim=0)
    for i in range(len(fg)):
        if torch.sum((fg[i] * overlap) > 1) > 0:
            pseudo_gt[pseudo_gt == i + 1] = -1

    entropy = -torch.sum(mask_prompt * torch.log(mask_prompt + 1e-10), dim=0)
    ignored_map = entropy < 0.3
    pseudo_gt[ignored_map != 1] = -1

    return pseudo_gt

def apply_coords_torch(coords: torch.Tensor, original_size, target_size) -> torch.Tensor:
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    from copy import deepcopy
    """
    Expects a torch tensor with length 2 in the last dimension. Requires the
    original image size in (H, W) format.
    """

    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(
        original_size[0], original_size[1], target_size
    )
    coords = deepcopy(coords).to(torch.float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords
def _iou_loss(pred, target, ignored_map=None):
    pred = torch.sigmoid(pred)
    if ignored_map == None:
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
    else:
        inter = ((pred * target) * ignored_map).sum(dim=(2, 3))
        union = ((pred + target) * ignored_map).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / (union + 1e-7))
    return iou.mean()

class SAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None, device=None):
        super().__init__()
        self.embed_dim = encoder_mode['embed_dim']
        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
        )
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.prompt_encoder = PromptEncoder(
            embed_dim=encoder_mode['prompt_embed_dim'],
            image_embedding_size=(inp_size//encoder_mode['patch_size'],inp_size//encoder_mode['patch_size']),
            input_image_size=(inp_size,inp_size),
            mask_in_chans=16
        )

        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.device = device

        if 'evp' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "prompt_encoder" not in k: #and "mask_decoder" not in k
                    p.requires_grad = False

        self.loss_mode = loss
        if self.loss_mode == 'mse':
            self.criterionBCE = torch.nn.MSELoss()

        elif self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss() # reduction='none' ,  pos_weight=torch.tensor(5)
            self.criterionIOU = IOU()

        # self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])

        # self.HQ_model = MaskDecoderHQ('vit_h')

        # HQ_module = self.HQ_model.modules
        # HQ_module.load_state_dict('pppp.pth')

    def set_input(self, input, gt_mask=None, clu=None, vor=None):
        self.input = input.to(self.device)
        if gt_mask is not None:
            self.gt_mask = gt_mask.to(self.device)
        if clu is not None:
            self.clu = clu.to(self.device)
            self.vor = vor.to(self.device)

    def forward(self):  # , point_prompt=None
        bs = len(self.input)

        self.features, self.interm_embeddings = self.image_encoder(self.input)

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        masks, _ = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            mask_token_only=True,
            local_path=False,
            interm_embeddings=self.interm_embeddings[0],
        )
        self.pred_mask = self.postprocess_masks(masks, self.inp_size, (224, 224))

        # return self.pred_mask, self.masks_hq #, self.pred_mask

    def backward_G(self):
        # train binary mask
        bce_loss = self.criterionBCE(self.pred_mask, self.gt_mask.float())

        # train pseudo label
        if self.loss_mode == 'iou':
            iou_loss = _iou_loss(self.pred_mask, self.gt_mask)
            self.loss_G = bce_loss + iou_loss
        else:
            self.loss_G = bce_loss

    def optimize_parameters(self, point_prompt=None, img_name=None):
        self.forward() #point_prompt
        self.backward_G()

        self.optimizer.zero_grad()  # set G's gradients to zero
        self.loss_G.backward()
        self.optimizer.step()  # udpate G's weights


    def infer(self, input, point_prompt=None):
        bs = 1

        self.features, self.interm_embeddings = self.image_encoder(input)

        # Embed prompts
        if point_prompt == None:
            sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size, self.image_embedding_size
            )
        else:
            # for b in range(point_prompt[0].shape[0]): 고려해야해... batch size...
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=point_prompt,
                boxes=None,
                masks=None,
            )


        masks, _ = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            mask_token_only=True,
            local_path=False,
            interm_embeddings=self.interm_embeddings[0],
        )
        self.pred_mask = self.postprocess_masks(masks, self.inp_size, (224,224))


        return self.pred_mask
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

