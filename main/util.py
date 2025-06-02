import torch
import torch.nn as nn
from omegaconf import OmegaConf
import importlib

def load_model(config_path: str, ckpt_path: str): # DONE
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    config = OmegaConf.load(f"{config_path}")
    model = instantiate_from_config(config.model)
    model.load_state_dict(state_dict, strict=False)

    model.cuda()
    model.eval()
    return model

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return _get_obj_from_str(config["target"])(**config.get("params", dict()))


def _get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_train_samples(opt, sample_data, sampling_step=None):
    """
    Prepare calibration samples. Always returns (xs, ts, conds),
    where `conds` is None if --cond is not set.
    """
    num_samples = 1024
    num_st = opt.cali_divide
    sampling_step = opt.ddim_steps if sampling_step is None else sampling_step

    if num_st == 1:
        xs = sample_data["xs"][0][:num_samples] if isinstance(sample_data, dict) else sample_data[:num_samples]
        ts = torch.ones(num_samples) * 800
    else:
        all_xs = sample_data["xs"]
        all_ts = sample_data["ts"]
        nsteps = len(all_ts)
        assert nsteps >= sampling_step, "Not enough timesteps for calibration"
        timesteps = list(range(0, nsteps, nsteps // num_st))
        xs = torch.cat([all_xs[i][:num_samples] for i in timesteps], dim=0)
        ts = torch.cat([all_ts[i][:num_samples] for i in timesteps], dim=0)

    conds = None
    if opt.cond:
        cs = sample_data["cs"]
        ucs = sample_data["ucs"]
        if num_st == 1:
            conds = cs[:num_samples]
        else:
            conds = torch.cat(
                [cs[i][:num_samples] for i in timesteps] + [ucs[i][:num_samples] for i in timesteps],
                dim=0
            )
    return xs, ts, conds

def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample
