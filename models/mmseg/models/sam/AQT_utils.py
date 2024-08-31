# ----------------------------------------------
# Created by Wei-Jie Huang
# ----------------------------------------------


import torch
from torch import nn
import torch.nn.functional as F


def remove_mask_and_warp(x, k, v, spatial_shapes=(28, 28)):
    """ Removes padding mask in sequence and warps each level of tokens into fixed-sized sequences.

    Args:
        src, pos (batch_size, sequence_length, d_model): patch tokens and position encodings
        padding_mask (batch_size, sequence_length): key padding mask
        level_start_index (num_feature_levels): start index of each feature level
        spatial_shapes (num_feature_levels, 2): spatial shape (H, W) of each feature level

    Returns:
        src_warped, pos_warped (batch_size, num_feature_levels, C, C): warped patch tokens and
        position encodings. The last two dimensions indicate sequence length (i.e., H*W) and model
        dimension, respectively.
    """
    B, H, W, C = x.shape
    k, v = k.view(B, H, W, C).permute(0, 3, 1, 2), v.view(B, H, W, C).permute(0, 3, 1, 2)
    k, v = F.adaptive_avg_pool2d(k, spatial_shapes), F.adaptive_avg_pool2d(v, spatial_shapes)
    k, v = k.flatten(2), v.flatten(2)
    return k, v

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DomainAttention(nn.Module):
    """ Wraps domain-adapting cross attention and MLP into a module.
        The operations are similar to those in Transformer, including normalization
        layers and dropout layers, while MLP is simplified as a linear layer.

    Args:
        d_model: total dimension of the model.
        n_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights.
    """

    def __init__(self, d_model, n_heads, dropout):
        super(DomainAttention, self).__init__()
        self.grl = GradientReversal()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    # def forward(self, query, src, pos=None, padding_mask=None):
    #     """ Args:
    #         query (batch_size, num_queries, d_model): discriminator query
    #         src, pos (batch_size, sequence_length, d_model): patch tokens and position encodings
    #         padding_mask (batch_size, sequence_length): key padding mask
    #     """
    #     r_query, _ = self.cross_attn(
    #         query=query.transpose(0, 1),
    #         key=self.grl(self.with_pos_embed(src, pos)).transpose(0, 1),
    #         value=self.grl(src).transpose(0, 1),
    #         key_padding_mask=padding_mask,
    #     )
    #     query = query + self.dropout1(r_query.transpose(0, 1))
    #     query = self.norm1(query)
    #     query = query + self.dropout2(self.linear(query))
    #     query = self.norm2(query)
    def forward(self, query, key, value, padding_mask=None):
        """ Args:
            query (batch_size, num_queries, d_model): discriminator query
            src, pos (batch_size, sequence_length, d_model): patch tokens and position encodings
            padding_mask (batch_size, sequence_length): key padding mask
        """
        r_query, _ = self.cross_attn(
            query=query.transpose(0, 1),
            key=self.grl(key).transpose(0, 1),
            value=self.grl(value).transpose(0, 1),
            key_padding_mask=padding_mask,
        )
        query = query + self.dropout1(r_query.transpose(0, 1))
        query = self.norm1(query)
        query = query + self.dropout2(self.linear(query))
        query = self.norm2(query)
        return query


# ------------------------------------------------------------------------------------------------------------------------------
# Copy-paste from https://github.com/jvanvugt/pytorch-domain-adaptation/blob/35ac3a5a04b5e1cf5b2145b6c442c2d678362eef/utils.py
# ------------------------------------------------------------------------------------------------------------------------------


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)