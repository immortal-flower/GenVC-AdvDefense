"""Shared one-time FLF2V model setup for persistent batch experiments."""

import re

import torch
from accelerate import dispatch_model, infer_auto_device_map

from sde_rf_wan.wan_flf2v_wrapper import WanFLF2VWrapper


def load_dual_gpu_flf2v_model(wan_ckpt, flow_shift=5.0,
                               gpu0_memory="22GiB", gpu1_memory="9GiB"):
    """Load and dispatch Wan FLF2V once, preserving the established 5090 layout."""
    model = WanFLF2VWrapper(
        wan_ckpt, config_name="flf2v-14B", flow_shift=flow_shift)
    print("\n[GVCC-BATCH] Loading 14B model once on CPU...")
    model.load("cpu", torch.bfloat16)

    print("[GVCC-BATCH] Dispatching DiT across both GPUs...")
    device_map = infer_auto_device_map(
        model.model, max_memory={0: gpu0_memory, 1: gpu1_memory})
    clean_map = {}
    for key, value in device_map.items():
        match = re.search(r"(^.*?blocks\.\d+)", key)
        if match:
            clean_map.setdefault(match.group(1), value)
        else:
            clean_map[key] = 0
    model.model = dispatch_model(model.model, device_map=clean_map)
    model.device = torch.device("cuda:0")

    if hasattr(model, "clip") and model.clip is not None:
        original_clip_visual = model.clip.visual

        def patched_clip_visual(videos, *args, **kwargs):
            with torch.no_grad():
                module = model.clip.model if hasattr(model.clip, "model") else model.clip
                module.to("cuda:1")
                values = ([v.to("cuda:1") for v in videos]
                          if isinstance(videos, list) else videos.to("cuda:1"))
                result = original_clip_visual(values, *args, **kwargs)
                module.to("cpu")
                torch.cuda.empty_cache()
                if hasattr(result, "to"):
                    return result.to("cuda:0")
                if isinstance(result, list):
                    return [item.to("cuda:0") for item in result]
                return result

        model.clip.visual = patched_clip_visual

    original_encode_prompt = model.encode_prompt

    def patched_encode_prompt(prompt, *args, **kwargs):
        with torch.no_grad():
            torch.cuda.empty_cache()
            previous_device = model.device
            model.device = torch.device("cuda:1")
            if hasattr(model, "text_encoder") and model.text_encoder is not None:
                module = (model.text_encoder.model
                          if hasattr(model.text_encoder, "model") else model.text_encoder)
                module.to("cuda:1")
            result = original_encode_prompt(prompt, *args, **kwargs)
            if hasattr(model, "text_encoder") and model.text_encoder is not None:
                module = (model.text_encoder.model
                          if hasattr(model.text_encoder, "model") else model.text_encoder)
                module.to("cpu")
            model.device = previous_device
            torch.cuda.empty_cache()
            if isinstance(result, dict):
                return {key: (value.to("cuda:0") if hasattr(value, "to") else value)
                        for key, value in result.items()}
            if hasattr(result, "to"):
                return result.to("cuda:0")
            if isinstance(result, list):
                return [value.to("cuda:0") if hasattr(value, "to") else value
                        for value in result]
            if isinstance(result, tuple):
                return tuple(value.to("cuda:0") if hasattr(value, "to") else value
                             for value in result)
            return result

    model.encode_prompt = patched_encode_prompt

    if hasattr(model, "vae") and model.vae is not None:
        original_vae_encode = model.vae.encode

        def patched_vae_encode(videos, *args, **kwargs):
            with torch.no_grad():
                torch.cuda.empty_cache()
                module = model.vae.model if hasattr(model.vae, "model") else model.vae
                module.to("cuda:1")
                values = ([v.to("cuda:1") for v in videos]
                          if isinstance(videos, list) else videos.to("cuda:1"))
                result = original_vae_encode(values, *args, **kwargs)
                module.to("cpu")
                torch.cuda.empty_cache()
                if isinstance(result, list):
                    return [item.to("cuda:0") for item in result]
                if isinstance(result, tuple):
                    return tuple(item.to("cuda:0") if hasattr(item, "to") else item
                                 for item in result)
                return result.to("cuda:0")

        model.vae.encode = patched_vae_encode
        original_vae_decode = model.vae.decode

        def patched_vae_decode(zs, *args, **kwargs):
            with torch.no_grad():
                torch.cuda.empty_cache()
                module = model.vae.model if hasattr(model.vae, "model") else model.vae
                module.to("cuda:1")
                values = ([z.to("cuda:1") for z in zs]
                          if isinstance(zs, list) else zs.to("cuda:1"))
                result = original_vae_decode(values, *args, **kwargs)
                module.to("cpu")
                torch.cuda.empty_cache()
                if isinstance(result, list):
                    return [item.to("cuda:0") for item in result]
                if isinstance(result, tuple):
                    return tuple(item.to("cuda:0") if hasattr(item, "to") else item
                                 for item in result)
                return result.to("cuda:0")

        model.vae.decode = patched_vae_decode

    model.model.cpu = lambda *args, **kwargs: model.model
    model.model.to = lambda *args, **kwargs: model.model
    print("[GVCC-BATCH] Persistent model ready; subsequent jobs reuse it.")
    return model
