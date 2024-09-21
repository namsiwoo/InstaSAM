# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import copy

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        #for HQ SAM
    def make_HQ_module(self, model_type, transformer_dim, num_token=1, HQ_transformer=None, local_transformer=None):
        vit_dim_dict = {"vit_b": 768, "medsam": 768, "vit_l": 1024, "vit_h": 1280}
        vit_dim = vit_dim_dict[model_type]

        self.HQ_transformer = copy.deepcopy(self.transformer)
        self.HQ_transformer.requires_grad_(True)

        self.mask_tokens2 = copy.deepcopy(self.mask_tokens)
        self.mask_tokens2.requires_grad_(True)

        self.hf_token = nn.Embedding(num_token, transformer_dim)  # num_embeddings:
        self.hf_token.requires_grad_(True)

        self.hf_mlp = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(num_token)
            ]
        )
        self.hf_mlp.requires_grad_(True)

        self.mask_mlp = copy.deepcopy(self.output_hypernetworks_mlps)
        self.mask_mlp.requires_grad_(True)

        self.output_upscaling_mask = copy.deepcopy(self.output_upscaling)
        self.output_upscaling_mask.requires_grad_(True)

        self.num_hq_token = num_token

        # self.compress_vit_feat = nn.Sequential(
        #     nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
        #     LayerNorm2d(transformer_dim),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        #
        # self.embedding_encoder = nn.Sequential(
        #     nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
        #     LayerNorm2d(transformer_dim // 4),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        # )
        #
        # self.embedding_maskfeature = nn.Sequential(
        #     nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
        #     LayerNorm2d(transformer_dim // 4),
        #     nn.GELU(),
        #     nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))

    def make_local_module(self, model_type, transformer_dim, local_transformer=None):
        import copy

        self.local_transformer = copy.deepcopy(self.transformer)
        self.local_transformer.requires_grad_(False)

        self.local_token = copy.deepcopy(self.mask_tokens)
        self.local_token.requires_grad_(False)

        self.local_mlp = copy.deepcopy(self.output_hypernetworks_mlps)
        self.local_mlp.requires_grad_(False)

        self.local_output_upscaling = copy.deepcopy(self.output_upscaling)
        self.local_output_upscaling.requires_grad_(False)


    # original SAM
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        mask_token_only: bool,
        local_path: bool,
        interm_embeddings,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """

        if mask_token_only == True:
            if local_path == True:
                masks, iou_preds = self.predict_masks_local( #, mask_feat
                    image_embeddings=image_embeddings,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_prompt_embeddings,
                    dense_prompt_embeddings=dense_prompt_embeddings,
                )
            else:
                masks, iou_preds = self.predict_masks(
                    image_embeddings=image_embeddings,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_prompt_embeddings,
                    dense_prompt_embeddings=dense_prompt_embeddings,
                )
        else:
            masks = []
            iou_preds = []
            for i_batch in range(len(image_embeddings)):
                mask, iou_pred, mask_feat = self.predict_masks_hq( #, mask_feat
                    image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch].unsqueeze(0),
                    dense_prompt_embeddings=dense_prompt_embeddings[i_batch].unsqueeze(0),
                    interm_embeddings = (interm_embeddings[i_batch].unsqueeze(0) if interm_embeddings is not None else None)
                )
                masks.append(mask)
                iou_preds.append(iou_pred)
            masks = torch.cat(masks, 0)
            iou_preds = torch.cat(iou_preds, 0)

        #print(masks.shape)

        # Select the correct mask or masks for outptu
        if multimask_output:
            mask_slice = slice(1, self.num_mask_tokens)
        else:
            mask_slice = slice(0, 1)
        masks_sam = masks[:, mask_slice, :, :]
        iou_preds = iou_preds[:, mask_slice]
        if mask_token_only == True:
            if local_path == True:
            # Prepare output
                return masks_sam, iou_preds#, mask_feat
            else:
                return masks_sam, iou_preds


        else:
            masks_hq = masks[:, slice(self.num_mask_tokens, self.num_mask_tokens + self.num_hq_token), :, :]
            return masks_sam, masks_hq, mask_feat

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

    def predict_masks_local(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:#, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.local_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.local_transformer(src, pos_src, tokens) ##############
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.local_output_upscaling(src) ############
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.local_mlp[i](mask_tokens_out[:, i, :])) ###########
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred#, image_embeddings

    def predict_masks_hq(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens

       # print(image_embeddings.shape, image_pe.shape, sparse_prompt_embeddings.shape, dense_prompt_embeddings.shape)

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens2.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        # print(image_embeddings.shape, tokens.shape, src.shape, dense_prompt_embeddings.shape)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        # hs, src = self.HQ_transformer(src, pos_src, tokens)

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens+self.num_hq_token), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        # upscaled_embedding = self.output_upscaling_mask(src)

        # vit_features = interm_embeddings.permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        # hq_feature = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)
        # upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding) + hq_feature


        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens+self.num_hq_token):
            if i<4:
                hyper_in_list.append(self.mask_mlp[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp[i-4](mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape

        # masks = (hyper_in[:, :4] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        # masks_hq = (hyper_in[:, 4:] @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        masks = (hyper_in[:, :4] @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        masks_hq = (hyper_in[:, 4:] @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        masks = torch.cat([masks,masks_hq],dim=1)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        # print(image_embeddings.shape)

        return masks, iou_pred, image_embeddings


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
