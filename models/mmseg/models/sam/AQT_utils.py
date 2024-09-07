# ----------------------------------------------
# Created by Wei-Jie Huang
# ----------------------------------------------


import torch
from torch import nn
import torch.nn.functional as F

class Domain_adapt(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        c_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        spatial_shape: int = 28,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.space_attn = DomainAttention(dim, num_heads, dropout=0.1)
        self.channel_attn = DomainAttention(c_dim, num_heads, dropout=0.1)
        self.spatial_shape = spatial_shape

        self.space_query = nn.Parameter(torch.randn(1, 1, dim))
        # self.space_query = nn.Embedding(1, 1, embed_dim)

        self.channel_query = nn.Linear(dim, 1)
        self.grl = GradientReversal()

        self.space_D = MLP(dim, dim, 1, 3)
        for layer in self.space_D.layers:
            nn.init.xavier_uniform_(layer.weight, gain=1)
            nn.init.constant_(layer.bias, 0)

        self.channel_D = MLP(c_dim, c_dim, 1, 3)
        for layer in self.channel_D.layers:
            nn.init.xavier_uniform_(layer.weight, gain=1)
            nn.init.constant_(layer.bias, 0)

        torch.nn.init.uniform_(self.qkv.weight)
        torch.nn.init.uniform_(self.qkv.bias)

    def make_query(self, x: torch.Tensor, space_query, channel_query) -> torch.Tensor:
        B, H, W, C = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        kv = self.qkv(x).reshape(B, H * W, 2, 1, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        k, v = kv.reshape(2, B, H * W, -1).unbind(0)
        space_query = self.space_attn(space_query, k, v)
        k, v = remove_mask_and_warp(x, k, v, self.spatial_shape)
        channel_query = self.channel_attn(channel_query, k, v)

        return space_query, channel_query

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):

        space_query = self.space_query.expand(x1[0].shape, -1, -1) # 1, 1, C
        channel_query = F.adaptive_avg_pool2d(x1[0].permute(0, 3, 1, 2), self.spatial_shape)
        channel_query = self.channel_query(self.grl(channel_query.flatten(2).transpose(1, 2))).transpose(1, 2) # 1, 1, L (L=H*W)

        channel_query2 = F.adaptive_avg_pool2d(x2[0].permute(0, 3, 1, 2), self.spatial_shape)
        channel_query2 = self.channel_query(self.grl(channel_query2.flatten(2).transpose(1, 2))).transpose(1, 2) # 1, 1, L (L=H*W)
        space_query2, = space_query.clone()

        for i in range(len(x1)):
            space_query, channel_query = self.make_query(x1[i], space_query, channel_query)
            space_query2, channel_query2 = self.make_query()(x2[i], space_query2, channel_query2)

        space_query = self.space_D(space_query)
        space_query2 = self.space_D(space_query2)
        channel_query = self.channel_D(channel_query)
        channel_query2 = self.channel_D(channel_query2)

        return (space_query, space_query2), (channel_query, channel_query2)

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