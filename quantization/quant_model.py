import torch.nn as nn

from quantization.quant_module import QuantModule, StraightThrough
from quantization.quant_block import QuantResBlock, QuantBasicTransformerBlock, ResBlock, BasicTransformerBlock, get_quantizables

class QuantModel(nn.Module):
    def __init__(self, u_net:nn.Module, quant_hyperparams:dict = (), **kwargs):
        super().__init__()
        self.model = u_net
        self.sm_abit = kwargs.get('sm_abit', 8)
        self.in_channels = u_net.in_channels
        self.quant_hyperparams = quant_hyperparams
        self.image_size = getattr(u_net, 'image_size', 32)
        self.quantizables = get_quantizables()
        
        # Refactor (replace) the module or block with Quant version
        print("Quant Module Refactor Start")
        self.quant_module_refactor(self.model, self.quant_hyperparams)
        print("Quant Block Refactor Start")
        self.quant_block_refactor(self.model, self.quant_hyperparams)

    def quant_module_refactor(self, module, quant_hyperparams):
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, quant_hyperparams))
            elif isinstance(child_module, StraightThrough): continue # StraightThrough used to indicate that this is the end
            else: self.quant_module_refactor(child_module, quant_hyperparams)

    def quant_block_refactor(self, block, quant_hyperparams):
        _, act_quant_params = quant_hyperparams
        for name, child_module in block.named_children():
            if type(child_module) in self.quantizables:
                if type(child_module) == BasicTransformerBlock:
                    setattr(block, name, QuantBasicTransformerBlock(child_module, act_quant_params, sm_abit=self.sm_abit))
                elif type(child_module) == ResBlock:
                    setattr(block, name, QuantResBlock(child_module, act_quant_params))
            else:
                self.quant_block_refactor(child_module, quant_hyperparams)

    def forward(self, x, timesteps=None, context=None):
        return self.model(x, timesteps, context)
    
    def set_running_stat(self, running_stat= bool, sm_only=False):
        for module in self.model.modules():
            if isinstance(module, QuantBasicTransformerBlock):
                if sm_only:
                    module.attn1.act_quantizer_w.running_stat = running_stat
                    module.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    module.attn1.act_quantizer_q.running_stat = running_stat
                    module.attn1.act_quantizer_k.running_stat = running_stat
                    module.attn1.act_quantizer_v.running_stat = running_stat
                    module.attn1.act_quantizer_w.running_stat = running_stat
                    module.attn2.act_quantizer_q.running_stat = running_stat
                    module.attn2.act_quantizer_k.running_stat = running_stat
                    module.attn2.act_quantizer_v.running_stat = running_stat
                    module.attn2.act_quantizer_w.running_stat = running_stat
            if isinstance(module, QuantModule) and not sm_only:
                module.set_running_stat(running_stat)

    def set_grad_ckpt(self, grad_ckpt: bool):
        for _, module in self.model.named_modules():
            if isinstance(module, (QuantBasicTransformerBlock, BasicTransformerBlock)):
                module.checkpoint = grad_ckpt