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

@register('sam')
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
            self.criterionBCE = torch.nn.BCEWithLogitsLoss(reduction='none') # reduction='none' ,  pos_weight=torch.tensor(5)
            self.criterionIOU = IOU()

        # self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])

        # self.HQ_model = MaskDecoderHQ('vit_h')

        # HQ_module = self.HQ_model.modules
        # HQ_module.load_state_dict('pppp.pth')
    def make_HQ_model(self, model_type, num_token):
        self.mask_decoder.make_HQ_module(model_type, self.prompt_embed_dim, num_token=num_token, #)
        HQ_transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=self.prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),)

        self.mask_decoder.make_local_module(model_type, self.prompt_embed_dim,
                                            local_transformer=TwoWayTransformer(
                                            depth=2,
                                            embedding_dim=self.prompt_embed_dim,
                                            mlp_dim=2048,
                                            num_heads=8,
                                            ), )


        if num_token == 2:
            self.criterionOFFSET = offset_Loss(224, 224)
            # self.criterionOFFSET = offset_Loss(300, 300)
        else:
            self.criterionOFFSET = offset_Loss_sonnet(224, 224)
    def make_adapter2(self):
        self.image_encoder.make_adapter2()

    def set_input(self, input, gt_mask=None, clu=None, vor=None):
        self.input = input.to(self.device)
        self.input = F.interpolate(self.input, (self.inp_size, self.inp_size), mode='bilinear', align_corners=True)
        if gt_mask is not None:
            self.gt_mask = gt_mask.to(self.device)
        if clu is not None:
            self.clu = clu.to(self.device)
            self.vor = vor.to(self.device)

    def forward_ssl(self, points=None, img_name=None, epoch=0): #, point_prompt=None
        bs = len(self.input)

        self.features, self.interm_embeddings, x_ori = self.image_encoder(self.input, mk_p_label=True)
        # x_ori = x_ori.detach()
        # del self.input

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        masks, masks_hq = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            mask_token_only=False,
            local_path=False,
            interm_embeddings=self.interm_embeddings[0], #.detach(),
        )
        self.pred_mask = self.postprocess_masks(masks, self.inp_size, (224,224))
        self.masks_hq = self.postprocess_masks(masks_hq, self.inp_size, (224,224))
        # self.pred_mask = self.postprocess_masks(masks, self.inp_size, (300,300))
        # self.masks_hq = self.postprocess_masks(masks_hq, self.inp_size, (300,300))

        del self.interm_embeddings, sparse_embeddings, dense_embeddings #self.features,

        # Make pseudo label using prompts
        pseudo_gt_local = torch.zeros_like(points.squeeze(1)).to(self.device)  # b, w, h (points.shape, b, 1, w, h)
        pseudo_gt_global = torch.zeros_like(points.squeeze(1)).to(self.device)  # b, w, h
        self.mask_prompt_adapter = []
        # if len(torch.unique(points[0]))>250:
        #     print(len(torch.unique(points[0])))
        for b in range(len(points)):
            # if len(torch.unique(points[0])) > 1:
            if torch.sum(points[b]) > 0:
                point_coord, point_label = make_point_prompt(points[b], only_fg=False)
                if len(torch.unique(points[0])) > 10000000: #cut
                    mask_prompt_adapter = torch.zeros_like(points.squeeze(1)).to(self.device).float()

                    #### Worse case....
                    # gt_local, gt_global = (torch.zeros(1, 224, 224).to(self.device)-1).float(), (torch.zeros(1, 224, 224).to(self.device)-1).float()

                    #### IDAL
                    gt_local, gt_global = torch.zeros(1, 224, 224).to(self.device), torch.zeros(1, 224, 224).to(self.device)
                    for num_p in range(0, torch.unique(points[b])[-1], 20):
                        if num_p == range(0, torch.sum(points[b]), 20)[-1]:
                            gt_local_part, gt_global_part, mask_prompt_adapter_part = self.make_pseudo_instance_map(b, (point_coord[num_p: ], point_label[num_p: ]), x_ori[b].unsqueeze(0))

                        else:
                            gt_local_part, gt_global_part, mask_prompt_adapter_part = self.make_pseudo_instance_map(b, (point_coord[num_p: num_p+20], point_label[num_p: num_p+20]), x_ori[b].unsqueeze(0))

                        gt_local_part = gt_local_part + num_p
                        gt_local_part[gt_local_part == num_p] = 0

                        gt_global_part = gt_global_part + num_p
                        gt_global_part[gt_global_part == num_p] = 0

                        gt_local = gt_local + gt_local_part
                        gt_global = gt_global + gt_global_part
                        if num_p == 0:
                            mask_prompt_adapter = mask_prompt_adapter_part.squeeze(1)
                        else:
                            mask_prompt_adapter = torch.cat([mask_prompt_adapter_part.squeeze(1), mask_prompt_adapter], dim=0)

                else:
                    # Make mask prompt using point labels
                    gt_local, gt_global, mask_prompt_adapter = self.make_pseudo_instance_map(b, (point_coord, point_label), x_ori[b].unsqueeze(0))


            else:
                gt_local = torch.zeros(1, 224, 224).to(self.device)
                gt_global, mask_prompt_adapter = self.make_pseudo_instance_map(b)
            self.mask_prompt_adapter.append(mask_prompt_adapter)
            pseudo_gt_local[b] = gt_local
            pseudo_gt_global[b] = gt_global

        del mask_prompt_adapter, points, x_ori#ignored_map

        if epoch>500:
            from utils.utils import accuracy_object_level, AJI_fast, save_checkpoint, load_checkpoint, mk_colored
            from PIL import Image

            # pseudo_gt2[0] = self.gt_mask

            aaa = mk_colored(pseudo_gt_local[0].detach().cpu().numpy()) *255
            aaa[pseudo_gt_local[0].detach().cpu().numpy()==0,:] = 0
            aaa[pseudo_gt_local[0].detach().cpu().numpy()==-1,:] = 255
            aaa = Image.fromarray(aaa.astype(np.uint8))
            aaa.save(img_name[:-4]+'_1local.png')

            aaa = mk_colored(pseudo_gt_global[0].detach().cpu().numpy()) *255
            aaa[pseudo_gt_global[0].detach().cpu().numpy()==0,:] = 0
            aaa[pseudo_gt_global[0].detach().cpu().numpy()==-1,:] = 255
            aaa = Image.fromarray(aaa.astype(np.uint8))
            aaa.save(img_name[:-4]+'_2global.png')

            # aaa = mk_colored(self.gt_mask[0].detach().cpu().numpy())*255
            # aaa = Image.fromarray(aaa.astype(np.uint8))
            # aaa.save(img_name[:-4]+'_3gt.png')

            aaa = (torch.sigmoid(self.pred_mask[0][0]).detach().cpu().numpy())*255
            aaa = Image.fromarray(aaa.astype(np.uint8))
            aaa.save(img_name[:-4]+'_6binary.png')

            import matplotlib.pyplot as plt
            # def colorize(ch, vmin=None, vmax=None):
            #     """Will clamp value value outside the provided range to vmax and vmin."""
            #     cmap = plt.get_cmap("jet")
            #     ch = np.squeeze(ch.astype("float32"))
            #     vmin = vmin if vmin is not None else ch.min()
            #     vmax = vmax if vmax is not None else ch.max()
            #     ch[ch > vmax] = vmax  # clamp value
            #     ch[ch < vmin] = vmin
            #     ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
            #     # take RGB from RGBA heat map
            #     ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
            #     return ch_cmap
            #
            # aaa = colorize(self.masks_hq[0, 0].detach().cpu().numpy())
            # aaa = Image.fromarray(aaa.astype(np.uint8))
            # aaa.save(img_name[:-4]+'_7h.png')
            #
            # aaa = colorize(self.masks_hq[0, 1].detach().cpu().numpy())
            # aaa = Image.fromarray(aaa.astype(np.uint8))
            # aaa.save(img_name[:-4]+'_8v.png')


            entropy = -torch.sum(torch.sigmoid(self.mask_prompt_adapter[b]) * torch.log(torch.sigmoid(self.mask_prompt_adapter[b]) + 1e-10), dim=0)
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(entropy[0].detach().cpu().numpy())
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(entropy[0].detach().cpu().numpy()<0.3)
            plt.savefig(img_name[:-4]+'_4entropy.png')
            #
            fg = (torch.sigmoid(self.mask_prompt_adapter[b]) > 0.5).float()
            overlap = torch.sum(fg, dim=0)
            plt.clf()
            plt.imshow(overlap[0].detach().cpu().numpy())
            plt.colorbar()
            plt.savefig(img_name[:-4]+'_5overlap.png')
        return pseudo_gt_local, pseudo_gt_global
        # return self.gt_mask

    def make_pseudo_instance_map(self, batch, point=None, ori_feature=None):
        if point == None:
            bs = len(self.input)
            sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.device)
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size, self.image_embedding_size
            )

        else:
            point_coord, point_label = point
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point_coord.to(self.device), point_label.to(self.device)),
                boxes=None,
                masks=None,
            )

        mask_prompt, iou_preds = self.mask_decoder(
            image_embeddings=self.features[batch].unsqueeze(0),  # self.features[b].unsqueeze(0)
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            mask_token_only=True,
            local_path=True,
            interm_embeddings=None, )
        mask_prompt_adapter = (self.postprocess_masks(mask_prompt, self.inp_size, (224, 224)))
        pseudo_gt_global = make_pseudo_gt(mask_prompt_adapter)

        if ori_feature != None:
            with torch.no_grad():
                mask_prompt, iou_preds = self.mask_decoder(
                    image_embeddings=ori_feature,  # self.features[b].unsqueeze(0)
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    mask_token_only=True,
                    local_path=True,
                    interm_embeddings=None,
                )
                mask_prompt_ori = self.postprocess_masks(mask_prompt, self.inp_size, (224, 224))  # b, 1 224, 224
                pseudo_gt_local = make_pseudo_gt(mask_prompt_ori)

            del sparse_embeddings, dense_embeddings, ori_feature, iou_preds, mask_prompt
            return pseudo_gt_local, pseudo_gt_global, mask_prompt_adapter
        else:
            del sparse_embeddings, dense_embeddings, iou_preds, mask_prompt
            return pseudo_gt_global, mask_prompt_adapter

    def backward_G_ssl(self, gt):
        binary_gt = gt.clone().unsqueeze(1)
        binary_gt[binary_gt > 0] = 1.
        ignored_map = (binary_gt != -1) #2, 224, 224
        gt[gt<0] = 0.

        bce_loss_sam = (self.criterionBCE(self.pred_mask, binary_gt.float())*ignored_map).mean()
        offset_loss, offset_gt = self.criterionOFFSET(self.masks_hq, gt, ignored_map=ignored_map)

        iou_loss_sam = _iou_loss(self.pred_mask, binary_gt, ignored_map=ignored_map)

        del binary_gt, ignored_map
        return bce_loss_sam, offset_loss, iou_loss_sam, offset_gt

    def backward_G_local(self, epoch, l_gt, g_gt):
        bce_loss_local = 0
        iou_loss_local = 0

        for b in range(len(self.mask_prompt_adapter)):
            train_map = (l_gt[b] != -1)
            train_map2 = (g_gt[b] != -1)
            l_pseudo_maks = torch.zeros_like(self.mask_prompt_adapter[b])
            g_pseudo_maks = torch.zeros_like(self.mask_prompt_adapter[b])
            for i in range(len(g_pseudo_maks)):
                l_pseudo_maks[i] = (l_gt[b].unsqueeze(0) == (i+1))
                g_pseudo_maks[i] = (g_gt[b].unsqueeze(0) == (i+1))

            bce_loss_local += (self.criterionBCE(self.mask_prompt_adapter[b], l_pseudo_maks)*train_map).mean()
            iou_loss_local += _iou_loss(self.mask_prompt_adapter[b].unsqueeze(0), l_pseudo_maks.unsqueeze(0), ignored_map=train_map)

            # bce_loss_local += (1-((epoch+1)/50))*(self.criterionBCE(self.mask_prompt_adapter[b], l_pseudo_maks)*train_map).mean()
            # iou_loss_local += (1-((epoch+1)/50))*_iou_loss(self.mask_prompt_adapter[b].unsqueeze(0), l_pseudo_maks.unsqueeze(0), ignored_map=train_map)
            # bce_loss_local += ((epoch+1)/50)*(self.criterionBCE(self.mask_prompt_adapter[b], g_pseudo_maks)*train_map2).mean()
            # iou_loss_local += ((epoch+1)/50)*_iou_loss(self.mask_prompt_adapter[b].unsqueeze(0), g_pseudo_maks.unsqueeze(0), ignored_map=train_map2)
        del l_pseudo_maks, g_pseudo_maks
        return bce_loss_local, iou_loss_local

    def backward_G_feature(self, epoch):
        for i in range(len(self.interm_embeddings)):
            print(self.interm_embeddings[i].shape)
        return bce_loss_local, iou_loss_local

    def forward(self):  # , point_prompt=None
        bs = len(self.input)

        self.features, self.interm_embeddings = self.image_encoder(self.input)
        # _, self.interm_embeddings, self.features = self.image_encoder(self.input, mk_p_label=True)

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        masks, masks_hq = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            mask_token_only=False,
            local_path=False,
            interm_embeddings=self.interm_embeddings[0],
        ) 
        self.pred_mask = self.postprocess_masks(masks, self.inp_size, (224, 224))
        self.masks_hq = self.postprocess_masks(masks_hq, self.inp_size, (224, 224))

        # self.pred_mask = self.postprocess_masks(masks, self.inp_size, (300, 300))
        # self.masks_hq = self.postprocess_masks(masks_hq, self.inp_size, (300, 300))

        # return self.pred_mask, self.masks_hq #, self.pred_mask

    def backward_G(self):
        binary_gt = self.gt_mask.clone().unsqueeze(1)
        binary_gt[binary_gt > 0] = 1.

        # train binary mask
        bce_loss = self.criterionBCE(self.pred_mask, binary_gt.float()).mean()

        # train offset mask
        offset_loss, offset_gt = self.criterionOFFSET(self.masks_hq, self.gt_mask)

        # train pseudo label
        if self.loss_mode == 'iou':
            iou_loss = _iou_loss(self.pred_mask, binary_gt)
            self.loss_G = offset_loss * 2 + bce_loss + iou_loss
        else:
            self.loss_G = offset_loss + bce_loss

        return bce_loss, offset_loss, iou_loss, offset_gt

    def optimize_parameters(self, point_prompt=None, img_name=None, semi=False, epoch=0):
        if point_prompt == None:
            self.forward() #point_prompt
            bce_loss, offset_loss, iou_loss, offset_gt = self.backward_G()  # calculate graidents for G
            self.loss_G = bce_loss + iou_loss + offset_loss
        else:
            if semi == False:
                local_gt, global_gt = self.forward_ssl(point_prompt, img_name, epoch)
                bce_loss, offset_loss, iou_loss, offset_gt = self.backward_G_ssl(global_gt)
                bce_loss_local, iou_loss_local = self.backward_G_local(epoch, local_gt, global_gt)
                self.loss_G = bce_loss + iou_loss + 5*offset_loss + bce_loss_local + iou_loss_local

            else:
                local_gt, global_gt = self.forward_ssl(point_prompt, img_name, epoch)
                self.backward_G_feature()
                # if img_name[-5] != '7': CoNSeP
                # print(img_name)
                # if img_name[-7:-4] == '2_3': #TNBC
                if img_name[-6:-4] == '_3': #MO, CPM
                    bce_loss, offset_loss, iou_loss, offset_gt = self.backward_G_ssl(self.gt_mask)
                    bce_loss_local, iou_loss_local = self.backward_G_local(epoch, self.gt_mask, global_gt)
                    if epoch< 10:
                        bce_loss, offset_loss, iou_loss, bce_loss_local, iou_loss_local = bce_loss * 200, offset_loss * 200, iou_loss * 200, bce_loss_local * 200, iou_loss_local * 200
                    else:
                        bce_loss, offset_loss, iou_loss, bce_loss_local, iou_loss_local = bce_loss*5, offset_loss*5, iou_loss*5, bce_loss_local*5, iou_loss_local*5
                # # elif epoch<10:
                # #     self.optimizer.zero_grad()
                # #     self.optimizer.step()
                # #     del self.input, self.gt_mask
                # #     return self.pred_mask, self.masks_hq, 0, 0, 0, self.masks_hq.clone(), 0, 0
                else:
                    bce_loss, offset_loss, iou_loss, offset_gt = self.backward_G()
                    bce_loss_local, iou_loss_local = self.backward_G_local(epoch, local_gt, global_gt)
                    # bce_loss_local, iou_loss_local = 0, 0
                self.loss_G = bce_loss + iou_loss + offset_loss + bce_loss_local + iou_loss_local

        del self.input
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.loss_G.backward()
        self.optimizer.step()  # udpate G's weights

        if point_prompt == None:
            return self.pred_mask, self.masks_hq, bce_loss.item(), offset_loss.item(), iou_loss.item(), offset_gt
        else:
            return self.pred_mask, self.masks_hq, bce_loss.item(), offset_loss.item(), iou_loss.item(), offset_gt, bce_loss_local, iou_loss_local

    def optimize_parameters_semi(self, point_prompt=None, img_name=None, semi=False, epoch=0):

        local_gt, global_gt = self.forward_ssl(point_prompt, img_name, epoch)
        # if img_name == 'train_4_5.png': #consep 04_7_0.png
        if img_name == '04_7_0.png': #tnbc
        # if img_name[-5] != '7': CoNSeP
        # if img_name[-7:-4] == '2_3': #TNBC
        # if img_name[-6:-4] == '_3': #MO, CPM
        # if img_name[-6:-4] == '_3' or img_name[-6:-4] == '11':  # MO, CPM
            bce_loss, offset_loss, iou_loss, offset_gt = self.backward_G_ssl(self.gt_mask)
            bce_loss_local, iou_loss_local = self.backward_G_local(epoch, self.gt_mask, global_gt)

            bce_loss, offset_loss, iou_loss, bce_loss_local, iou_loss_local = bce_loss*5, offset_loss*5, iou_loss*5, bce_loss_local*5, iou_loss_local*5

            del self.input
            self.loss_G = bce_loss + iou_loss + offset_loss + bce_loss_local + iou_loss_local
            self.optimizer.zero_grad()  # set G's gradients to zero
            self.loss_G.backward()
            self.optimizer.step()  # udpate G's weights
            return self.pred_mask, self.masks_hq, bce_loss.item(), offset_loss.item(), iou_loss.item(), offset_gt, bce_loss_local, iou_loss_local

        else:
            if epoch < 15:
                return self.pred_mask, self.masks_hq, 0, 0, 0, self.masks_hq, 0, 0
            else:
                bce_loss, offset_loss, iou_loss, offset_gt = self.backward_G_ssl(global_gt)
                bce_loss_local, iou_loss_local = 0, 0
                self.loss_G = bce_loss + iou_loss + offset_loss + bce_loss_local + iou_loss_local
                self.optimizer.zero_grad()  # set G's gradients to zero
                self.loss_G.backward()
                self.optimizer.step()  # udpate G's weights
                return self.pred_mask, self.masks_hq, bce_loss.item(), offset_loss.item(), iou_loss.item(), offset_gt, bce_loss_local, iou_loss_local




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


        masks, masks_hq = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            mask_token_only=False,
            local_path=False,
            interm_embeddings=self.interm_embeddings[0],
        )
        self.pred_mask = self.postprocess_masks(masks, self.inp_size, (224,224))
        self.masks_hq = self.postprocess_masks(masks_hq, self.inp_size, (224,224))

        # self.pred_mask = self.postprocess_masks(masks, self.inp_size, (300,300))
        # self.masks_hq = self.postprocess_masks(masks_hq, self.inp_size, (300,300))


        return self.pred_mask, self.masks_hq #, self.pred_mask
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

