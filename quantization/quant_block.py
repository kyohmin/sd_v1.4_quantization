import torch
import torch.nn as nn
from torch import einsum
from types import MethodType
from einops import rearrange, repeat


from quantization.quantizer import Quantizer
from quantization.quant_module import QuantModule, StraightThrough

from ldm.modules.diffusionmodules.openaimodel import TimestepBlock, ResBlock, checkpoint
from ldm.modules.attention import BasicTransformerBlock, exists, default

class BaseQuantBlock(nn.Module):
    def __init__(self, act_quant_params: dict={}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False

        self.act_quantizer = Quantizer(**act_quant_params)
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

        def set_quant_state(self, weight_quant:bool = False, act_quant: bool = False):
            self.use_weight_quant = weight_quant
            self.use_act_quant = act_quant


class QuantResBlock(BaseQuantBlock, TimestepBlock): # TimestepBlock to ensure module to receive emb
    def __init__(self, res = ResBlock, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        # Load params from original ResBlock res
        self.channels = res.channels
        self.emb_channels = res.emb_channels
        self.dropout = res.dropout
        self.out_channels = res.out_channels
        self.use_conv = res.use_conv
        self.use_checkpoint = res.use_checkpoint
        self.use_scale_shift_norm = res.use_scale_shift_norm

        self.in_layers = res.in_layers
        self.updown = res.updown
        self.h_upd = res.h_upd
        self.x_upd = res.x_upd

        self.emb_layers = res.emb_layers
        self.out_layers = res.out_layers

        self.skip_connection = res.skip_connection

    def forward(self, x, emb=None, split=0):
        if split != 0 and self.skip_connection == 0: # If split = True
            return checkpoint(self._forward, (x, emb, split), self.parameters(), self.use_checkpoint) 
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)  
    
    def _forward(self, x, emb, split=0):
        if emb is None:
            assert(len(x) == 2)
            x, emb = x
        assert x.shape[2] == x.shape[3] # Ensures H == W

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shpe) < len(h.shape): emb_out = emb_out[..., None] # Match the shape until it is.
        
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:] # Norm & (SiLU, DropOut, Zero_module)
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        if split != 0: return self.skip_connection(x, split=split) + h
        return self.skip_connection(x) + h


def cross_attn_forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    if self.use_act_quant:
        quant_q = self.act_quantizer_q(q)
        quant_k = self.act_quantizer_k(k)
        sim = einsum('b i d, b j d -> b i j', quant_q, quant_k) * self.scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)

    if self.use_act_quant:
        out = einsum('b i j, b j d -> b i d', self.act_quantizer_w(attn), self.act_quantizer_v(v))
    else:
        out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


class QuantBasicTransformerBlock(BaseQuantBlock):
    def __init__(self, tran: BasicTransformerBlock, act_quant_params: dict = {}, sm_abit: int = 8):
        super().__init__(act_quant_params)
        self.attn1 = tran.attn1
        self.ff = tran.ff
        self.attn2 = tran.attn2
        
        self.norm1 = tran.norm1
        self.norm2 = tran.norm2
        self.norm3 = tran.norm3
        self.checkpoint = tran.checkpoint
        # self.checkpoint = False

        # logger.info(f"quant attn matmul")
        self.attn1.act_quantizer_q = Quantizer(**act_quant_params)
        self.attn1.act_quantizer_k = Quantizer(**act_quant_params)
        self.attn1.act_quantizer_v = Quantizer(**act_quant_params)

        self.attn2.act_quantizer_q = Quantizer(**act_quant_params)
        self.attn2.act_quantizer_k = Quantizer(**act_quant_params)
        self.attn2.act_quantizer_v = Quantizer(**act_quant_params)
        
        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        act_quant_params_w['always_zero'] = False
        self.attn1.act_quantizer_w = Quantizer(**act_quant_params_w)
        self.attn2.act_quantizer_w = Quantizer(**act_quant_params_w)

        self.attn1.forward = MethodType(cross_attn_forward, self.attn1)
        self.attn2.forward = MethodType(cross_attn_forward, self.attn2)
        self.attn1.use_act_quant = False
        self.attn2.use_act_quant = False

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        if context is None:
            assert(len(x) == 2)
            x, context = x

        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.attn1.use_act_quant = act_quant
        self.attn2.use_act_quant = act_quant

        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)

def get_quantizables():
    quantizables = {
        ResBlock: QuantResBlock,
        BasicTransformerBlock: QuantBasicTransformerBlock
    }
    
    return quantizables