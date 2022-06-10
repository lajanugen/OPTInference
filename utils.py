import json
import os
from transformers import OPTConfig
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map, load_checkpoint_in_model
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_opt_model(weights_path, fp16=False):

    # Load config
    with open(os.path.join(weights_path, 'config.json'), 'r') as f:
        config = json.load(f)
    config["_name_or_path"] = weights_path
    config = OPTConfig(**config)

    # Initializes an empty shell with the model. This is instant and does not take any RAM.
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    # Initialize the model under the previous context manager breaks the tied weights.
    model.tie_weights()

    # Infer device map automatically
    if fp16:
        device_map = infer_auto_device_map(model.model, no_split_module_classes=["OPTDecoderLayer"], dtype='float16')
    else:
        device_map = infer_auto_device_map(model.model, no_split_module_classes=["OPTDecoderLayer"])

    # Load weights
    load_checkpoint_in_model(
        model.model, 
        weights_path, 
        device_map=device_map,
        offload_folder=None, 
        dtype='float16' if fp16 else 'float32',
        offload_state_dict=True
    )
    model.tie_weights()

    # Without this part, torch complains about tensors being in different devices
    full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
    full_model_device_map["lm_head"] = 0
    dispatch_model(model, device_map=full_model_device_map)
    
    return model