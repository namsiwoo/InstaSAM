# ----------------------------------------------
# Created by Wei-Jie Huang
# ----------------------------------------------


import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init

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

    def make_query(self, x: torch.Tensor, space_query, channel_query):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        kv = self.qkv(x).reshape(B, H * W, 2, 1, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        k, v = kv.reshape(2, B, H * W, -1).unbind(0)
        space_query = self.space_attn(space_query, k, v)
        k, v = remove_mask_and_warp(x, k, v, self.spatial_shape)
        channel_query = self.channel_attn(channel_query, k, v)

        return space_query, channel_query

    def forward(self, x1, x2):
        # print(len(x1), len(x1[0]))
        # print((x1[0].shape))

        space_query = self.space_query.expand(x1[0].shape[0], -1, -1) # 1, 1, C
        space_query2 = space_query.clone()

        channel_query = F.adaptive_avg_pool2d(x1[0].permute(0, 3, 1, 2), self.spatial_shape)
        channel_query = self.channel_query(self.grl(channel_query.flatten(2).transpose(1, 2))).transpose(1, 2) # 1, 1, L (L=H*W)
        channel_query2 = F.adaptive_avg_pool2d(x2[0].permute(0, 3, 1, 2), self.spatial_shape)
        channel_query2 = self.channel_query(self.grl(channel_query2.flatten(2).transpose(1, 2))).transpose(1, 2) # 1, 1, L (L=H*W)


        for i in range(len(x1)):
            space_query, channel_query = self.make_query(x1[i], space_query, channel_query)
            space_query2, channel_query2 = self.make_query(x2[i], space_query2, channel_query2)

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
############################################

class Discriminator(nn.Module):
    def __init__(self, input_nc=3, num_filters_last=64, n_layers=3):
        super(Discriminator, self).__init__()

        layers = [nn.Conv2d(input_nc, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# class Discriminator(nn.Module):
#     def __init__(self, input_nc=3, ndf=64, netD="n_layers", n_layers_D=3, norm='batch'):
#         super(Discriminator, self).__init__()
#         self.model = define_D(input_nc=input_nc, ndf=ndf, netD=netD, n_layers_D=n_layers_D, norm=norm,
#                               init_type='normal', init_gain=0.02)
#
#     def forward(self, input):
#         return self.model(input)
# def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
#     """Create a discriminator
#     Parameters:
#         input_nc (int)     -- the number of channels in input images
#         ndf (int)          -- the number of filters in the first conv layer
#         netD (str)         -- the architecture's name: basic | n_layers | pixel
#         n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
#         norm (str)         -- the type of normalization layers used in the network.
#         init_type (str)    -- the name of the initialization method.
#         init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
#         gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
#     Returns a discriminator
#     Our current implementation provides three types of discriminators:
#         [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
#         It can classify whether 70×70 overlapping patches are real or fake.
#         Such a patch-level discriminator architecture has fewer parameters
#         than a full-image discriminator and can work on arbitrarily-sized images
#         in a fully convolutional fashion.
#         [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
#         with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
#         [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
#         It encourages greater color diversity but has no effect on spatial statistics.
#     The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
#     """
#     net = None
#     norm_layer = get_norm_layer(norm_type=norm)
#
#     if netD == 'basic':  # default PatchGAN classifier
#         net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
#     elif netD == 'n_layers':  # more options
#         net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
#     elif netD == 'pixel':  # classify if each pixel is real or fake
#         net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
#     else:
#         raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
#     return init_net(net, init_type, init_gain, gpu_ids)
#
# def get_norm_layer(norm_type='instance'):
#     """Return a normalization layer
#     Parameters:
#         norm_type (str) -- the name of the normalization layer: batch | instance | none
#     For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
#     For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
#     """
#     if norm_type == 'batch':
#         norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
#     elif norm_type == 'instance':
#         norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
#     elif norm_type == 'none':
#         def norm_layer(x):
#             return Identity()
#     else:
#         raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
#     return norm_layer
#
# class NLayerDiscriminator(nn.Module):
#     """Defines a PatchGAN discriminator"""
#
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
#         """Construct a PatchGAN discriminator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(NLayerDiscriminator, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         kw = 4
#         padw = 1
#         sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]
#
#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         sequence += [
#             nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
#
#         self.model = nn.Sequential(*sequence)
#
#     def forward(self, input):
#         """Standard forward."""
#         return self.model(input)
#
# def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
#     """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
#     Parameters:
#         net (network)      -- the network to be initialized
#         init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
#         gain (float)       -- scaling factor for normal, xavier and orthogonal.
#         gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
#     Return an initialized network.
#     """
#     if len(gpu_ids) > 0:
#         assert (torch.cuda.is_available())
#         net.to(gpu_ids[0])
#         net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
#     init_weights(net, init_type, init_gain=init_gain)
#     return net
#
# def init_weights(net, init_type='normal', init_gain=0.02):
#     """Initialize network weights.
#     Parameters:
#         net (network)   -- network to be initialized
#         init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
#         init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
#     We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
#     work better for some applications. Feel free to try yourself.
#     """
#
#     def init_func(m):  # define the initialization function
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if init_type == 'normal':
#                 init.normal_(m.weight.data, 0.0, init_gain)
#             elif init_type == 'xavier':
#                 init.xavier_normal_(m.weight.data, gain=init_gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal_(m.weight.data, gain=init_gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find(
#                 'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
#             init.normal_(m.weight.data, 1.0, init_gain)
#             init.constant_(m.bias.data, 0.0)
#
#     print('initialize network with %s' % init_type)
#     net.apply(init_func)  # apply the initialization function <init_func>