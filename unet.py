import math
from abc import abstractmethod

import numpy as np

import torch
from torch import einsum, nn

from einops import rearrange

from stable_diffusion.ldm.modules.diffusionmodules.util import (
    conv_nd,
    normalization,
    timestep_embedding,
    zero_module,
)

from stable_diffusion.ldm.modules.attention import (
    FeedForward,
    Normalize,
)

from stable_diffusion.ldm.modules.diffusionmodules.openaimodel import (
    convert_module_to_f16,
    convert_module_to_f32,
    QKVAttention,
    QKVAttentionLegacy,
    Upsample,
    Downsample,
)


# https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/util.py#L102
# CheckpointFunction が書き変わっているため．
def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


# https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/util.py#L119
# backward の最初に float() が追加されている．
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.float().detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


# https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L152
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        from stable_diffusion.ldm.modules.attention import default
        
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        from stable_diffusion.ldm.modules.attention import exists, default
        
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


# https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L196
# checkpoint が書き変わっているため．
class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention for the image
        self.attnc = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention for the context
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
    
    def _forward(self, x, context=None):
        
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context, mask=None) + x
        x = self.ff(self.norm3(x)) + x
        return x


# https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L218
class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, part='encoder', vocab_size=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
        self.part = part
    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        #print('x spatial trans in', x.shape)
        
        
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        if self.part != 'sca':
            x = rearrange(x, 'b c h w -> b (h w) c')
    
        for block in self.transformer_blocks:
            x = block(x, context=context)
        if self.part != 'sca':
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


# https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/openaimodel.py#L62
# forward の引数に context が追加されているが，特に使ってなさそう．
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, context):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


# https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/openaimodel.py#L74
# TimestepBlock が書き変わっているため．
# しかし，そもそも呼んでいないのでは（呼んでいるなら context が足りない）？
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
                
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
                
            else:
                x = layer(x)
        return x


# https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/openaimodel.py#L163
# TimestepBlock が書き変わっているため．
class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
            
        # if context is None:
        #     context= torch.zeros(emb.shape).to(emb.device)
        
        # emb = torch.cat([emb, context], dim=-1)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
    
    
# https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/openaimodel.py#L278
# qkv, proj_out の conv の次元が違う（意図的なのか？）．
class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


##################################################################################


# https://github.com/koninik/WordStylist/blob/f18522306e533a01eb823dc4369a4bcb7ea67bcc/unet.py#L688
class CharAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CharAttention, self).__init__()
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
        :param x: (batch_size, sequence_length, input_size)
        """
        
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # Calculate attention scores
        scores = query @ key.transpose(-2, -1)
        scores = self.softmax(scores)
        
        # Calculate weighted sum of the values
        rads_embs = scores @ value
        return rads_embs


# https://github.com/koninik/WordStylist/blob/f18522306e533a01eb823dc4369a4bcb7ea67bcc/unet.py#L711
class CharEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, char_length):
        super(CharEncoder, self).__init__()
        
        self.embedding_dim = hidden_size
        self.char_length = char_length
        
        self.embedding = nn.Embedding(input_size, hidden_size, max_norm=1)
        self.attention = CharAttention(hidden_size, hidden_size)
        
        # 部首埋め込み (正規化されている) に掛ける係数
        self.rad_emb_norm = 1
        
        # 位置埋め込みの精度
        self.pos_enc_precision = 1000
        
        # 位置埋め込みの前計算
        self.pos_enc_list = [] # [i][p * k] -> encoding tensor
        
        pos_enc_type = 4
        print(f"{pos_enc_type=}")
        
        if pos_enc_type == 0:
            self.rad_emb_norm = ((self.embedding_dim / 2) ** 0.5) * 4 # 埋め込みのノルムの分
            
            for i in range(4):
                T = 1000 * (i + 1)
                
                self.pos_enc_list.append([])
                for p in np.linspace(0, 1, self.pos_enc_precision):
                    self.pos_enc_list[-1].append(torch.zeros(self.embedding_dim))

                    for i in range(0, self.embedding_dim, 2):
                        theta = (p * self.embedding_dim) / (T ** (i / self.embedding_dim))
                        self.pos_enc_list[-1][-1][i] = math.sin(theta)
                        self.pos_enc_list[-1][-1][i + 1] = math.cos(theta)

                    assert (
                        abs(torch.norm(self.pos_enc_list[-1][-1], 2) - self.rad_emb_norm / 4)
                        < 1e-4
                    )
        
        elif pos_enc_type == 1:
            assert self.embedding_dim % 12 == 0
            
            self.rad_emb_norm = ((self.embedding_dim / 2) ** 0.5) * 4 # 埋め込みのノルムの分
            
            for _ in range(4):
                T = 1000
                
                self.pos_enc_list.append([])
                for p in np.linspace(0, 1, self.pos_enc_precision):
                    self.pos_enc_list[-1].append(torch.zeros(self.embedding_dim))
                    
                    for i in range(0, self.embedding_dim, 12):
                        theta = (p * self.embedding_dim) / (T ** (i / self.embedding_dim))
                        
                        self.pos_enc_list[-1][-1][i + 0] = math.sin(theta)
                        self.pos_enc_list[-1][-1][i + 1] = math.sin(theta)
                        self.pos_enc_list[-1][-1][i + 2] = math.sin(theta)
                        self.pos_enc_list[-1][-1][i + 3] = math.sin(theta)
                        self.pos_enc_list[-1][-1][i + 4] = math.sin(theta)
                        self.pos_enc_list[-1][-1][i + 5] = math.sin(theta)
                        
                        self.pos_enc_list[-1][-1][i + 6] = math.cos(theta)
                        self.pos_enc_list[-1][-1][i + 7] = math.cos(theta)
                        self.pos_enc_list[-1][-1][i + 8] = math.cos(theta)
                        self.pos_enc_list[-1][-1][i + 9] = math.cos(theta)
                        self.pos_enc_list[-1][-1][i + 10] = math.cos(theta)
                        self.pos_enc_list[-1][-1][i + 11] = math.cos(theta)
                    
                    assert (
                        abs(torch.norm(self.pos_enc_list[-1][-1], 2) - self.rad_emb_norm / 4)
                        < 1e-4
                    )
                
            for i_p in range(len(self.pos_enc_list[0])):
                for i in range(0, self.embedding_dim, 6):
                    self.pos_enc_list[0][i_p][i + 0] *= -1
                    self.pos_enc_list[1][i_p][i + 0] *= -1
                    
                    self.pos_enc_list[0][i_p][i + 1] *= -1
                    self.pos_enc_list[2][i_p][i + 1] *= -1
                    
                    self.pos_enc_list[0][i_p][i + 2] *= -1
                    self.pos_enc_list[3][i_p][i + 2] *= -1
                    
                    self.pos_enc_list[1][i_p][i + 3] *= -1
                    self.pos_enc_list[2][i_p][i + 3] *= -1
                    
                    self.pos_enc_list[1][i_p][i + 4] *= -1
                    self.pos_enc_list[3][i_p][i + 4] *= -1
                    
                    self.pos_enc_list[2][i_p][i + 5] *= -1
                    self.pos_enc_list[3][i_p][i + 5] *= -1
                    
        elif pos_enc_type == 2:
            assert self.embedding_dim % 4 == 0
            
            self.rad_emb_norm = ((self.embedding_dim / 2) ** 0.5) * 4 # 埋め込みのノルムの分
            
            for _ in range(4):
                T = 1000
                
                self.pos_enc_list.append([])
                for p in np.linspace(0, 1, self.pos_enc_precision):
                    self.pos_enc_list[-1].append(torch.zeros(self.embedding_dim))
                    
                    for i in range(0, self.embedding_dim, 4):
                        theta = (p * self.embedding_dim) / (T ** (i / self.embedding_dim))
                        
                        self.pos_enc_list[-1][-1][i + 0] = math.sin(theta)
                        self.pos_enc_list[-1][-1][i + 1] = math.cos(theta)
                        self.pos_enc_list[-1][-1][i + 2] = math.sin(theta)
                        self.pos_enc_list[-1][-1][i + 3] = math.cos(theta)
                    
                    assert (
                        abs(torch.norm(self.pos_enc_list[-1][-1], 2) - self.rad_emb_norm / 4)
                        < 1e-4
                    )
                
            for i_p in range(len(self.pos_enc_list[0])):
                for i in range(0, self.embedding_dim, 4):
                    self.pos_enc_list[0][i_p][i + 0] *= -1
                    self.pos_enc_list[1][i_p][i + 1] *= -1
                    self.pos_enc_list[2][i_p][i + 2] *= -1
                    self.pos_enc_list[3][i_p][i + 3] *= -1
                    
        elif pos_enc_type == 3:
            assert self.embedding_dim % 8 == 0
            
            self.rad_emb_norm = ((self.embedding_dim / 8) ** 0.5) * 4 # 埋め込みのノルムの分
            
            k = self.embedding_dim // 4
            for d in range(4):
                T = 1000
                
                self.pos_enc_list.append([])
                for p in np.linspace(0, 1, self.pos_enc_precision):
                    self.pos_enc_list[-1].append(torch.zeros(self.embedding_dim))
                    
                    i_0 = d * k
                    for i in range(0, k, 2):
                        theta = p / (T ** (i / k))
                        
                        self.pos_enc_list[-1][-1][i_0 + i + 0] = math.sin(theta)
                        self.pos_enc_list[-1][-1][i_0 + i + 1] = math.cos(theta)
                    
                    assert (
                        abs(torch.norm(self.pos_enc_list[-1][-1], 2) - self.rad_emb_norm / 4)
                        < 1e-4
                    )
                    
        elif pos_enc_type == 4:
            assert self.embedding_dim % 8 == 0
            
            self.pos_enc_precision = self.embedding_dim - 1 # 位置埋め込みの精度
            self.rad_emb_norm = ((self.embedding_dim / 8) ** 0.5) * 4 # 埋め込みのノルムの分
            
            k = self.embedding_dim // 4
            for d in range(4):
                T = 1000
                
                self.pos_enc_list.append([])
                for p in np.linspace(0, 1, self.pos_enc_precision):
                    self.pos_enc_list[-1].append(torch.zeros(self.embedding_dim))
                    
                    i_0 = d * k
                    for i in range(0, k, 2):
                        theta = p / (T ** (i / k))
                        
                        self.pos_enc_list[-1][-1][i_0 + i + 0] = math.sin(theta)
                        self.pos_enc_list[-1][-1][i_0 + i + 1] = math.cos(theta)
                    
                    assert (
                        abs(torch.norm(self.pos_enc_list[-1][-1], 2) - self.rad_emb_norm / 4)
                        < 1e-4
                    )
    
    def forward(self, chars):
        """
        :param x: list[list[Radical]], length: batch_size
        """
        device = self.embedding.weight.device
        
        # バッチ毎にひとつひとつ計算 (最適化したい)
        rads_embs = []
        for char in chars:
            rad_embs = []
            
            for radical in char.radicals:
                idx = torch.tensor([radical.idx], device=device)
                
                rad_emb = self.embedding(idx)
                rad_emb *= self.rad_emb_norm
                
                for i, p in enumerate((radical.center_x, radical.center_y, radical.width, radical.height)):
                    encoding = self.pos_enc_list[i][int(p * self.pos_enc_precision)].to(device)
                    rad_emb += encoding
                
                rad_embs.append(rad_emb)
                
            while len(rad_embs) < self.char_length:
                rad_embs.append(torch.zeros(rad_embs[0].shape, device=device))
            
            rad_embs = torch.cat(rad_embs, dim=0)
            rads_embs.append(rad_embs)
        
        rads_embs = torch.stack(rads_embs)
        rads_embs = self.attention(rads_embs)
        return rads_embs # (batch_size, char_length, hidden_size)


##################################################################################


# https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/openaimodel.py#L413
class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        # custom
        use_spatial_transformer=True,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=768,                 # custom transformer support
        vocab_size=256,                  # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=False,
        char_length=-1,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.char_encoder = CharEncoder(vocab_size, context_dim, char_length)
        
        # ==================== INPUT BLOCK ====================
        
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        # ==================== MIDDLE BLOCK ====================
        
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # ==================== OUTPUT BLOCK ====================
        
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            nn.Conv2d(model_channels, n_embed, 1),
            nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )
        
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

        
    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
    
    def forward(self, x, timesteps=None, chars=None, writers_idx=None, mix_rate=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param chars: conditioning plugged in via crossattn
        :param writers_idx: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        
        emb = self.time_embed(t_emb)
        
        if self.num_classes is None:
            assert writers_idx is None
        
        else:
            assert writers_idx is not None
            assert writers_idx.shape == (x.shape[0],)
            emb = emb + self.label_emb(writers_idx)
        
        chars = self.char_encoder(chars)
        
        h = x.type(self.dtype)
        
        # INPUT BLOCKS
        for module in self.input_blocks:
            h = module(h, emb, chars)
            hs.append(h)
        
        # MIDDLE BLOCK
        h = self.middle_block(h, emb, chars)
        
        # OUTPUT BLOCKS
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, chars)
            
        h = h.type(x.dtype)
        
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
