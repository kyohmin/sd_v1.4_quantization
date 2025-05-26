import torch
import importlib

def load_model(config, ckpt):
    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    if "target" not in config["model"]:
        if config["model"] in ("__is_first_stage__", "__is_unconditional__"):
            return None
        raise KeyError("Expected key 'target' to instantiate.")
    
    target = config["model"]["target"]
    params = config["model"].get("params", {})

    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    model = model_class(**params)

    model.load_state_dict(state_dict, strict=False)

    return model.cuda().eval()
