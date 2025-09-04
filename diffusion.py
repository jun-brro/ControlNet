import os
import argparse
import random
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import torch
import einops

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def _ensure_divisible_by(x: int, div: int) -> int:
    r = x % div
    return x if r == 0 else x - r


def _read_image(path: str, target_size: Optional[int] = None) -> np.ndarray:
    assert os.path.isfile(path), f"Input image not found: {path}"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size is not None and target_size > 0:
        h, w, _ = img.shape
        scale = float(target_size) / float(max(h, w))
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    h, w, _ = img.shape
    nh = _ensure_divisible_by(h, 8)
    nw = _ensure_divisible_by(w, 8)
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return img


def _read_mask(path: str, hw: Tuple[int, int]) -> np.ndarray:

    assert os.path.isfile(path), f"Mask image not found: {path}"
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    mask = cv2.resize(mask, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask.astype(np.float32) / 255.0)
    mask = np.clip(mask, 0.0, 1.0)
    mask = (mask > 0.5).astype(np.float32)
    return mask


def _make_soft_masks(mask_inside: np.ndarray, blur_sigma: float = 3.0, dilate_iter: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
      - inside_soft in [0,1]
      - keep_soft   in [0,1] where 1=keep (outside)
    """
    h, w = mask_inside.shape
    if dilate_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(mask_inside, k, iterations=int(dilate_iter))
    else:
        dilated = mask_inside
    if blur_sigma and blur_sigma > 0:
        inside_soft = cv2.GaussianBlur(dilated, ksize=(0, 0), sigmaX=blur_sigma)
    else:
        inside_soft = dilated
    inside_soft = np.clip(inside_soft, 0.0, 1.0).astype(np.float32)
    keep_soft = (1.0 - inside_soft).astype(np.float32)
    return inside_soft, keep_soft


def _to_torch_image(img_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    x = einops.rearrange(x, 'h w c -> 1 c h w').contiguous()
    x = x * 2.0 - 1.0
    if device.type == 'cuda':
        try:
            x = x.pin_memory()
        except Exception:
            pass
        return x.to(device, non_blocking=True)
    return x.to(device)


def _to_torch_map(arr_hw: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(arr_hw.astype(np.float32))
    t = t[None, None, ...].contiguous()
    if device.type == 'cuda':
        try:
            t = t.pin_memory()
        except Exception:
            pass
        return t.to(device, non_blocking=True)
    return t.to(device)


def _downsample_mask(mask_hw: np.ndarray, latent_hw: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask_hw, (latent_hw[1], latent_hw[0]), interpolation=cv2.INTER_AREA).astype(np.float32)


def _parse_device_map_arg(arg: Optional[str]) -> Optional[Dict[str, str]]:
    if not arg:
        return None
    mapping = {}
    parts = [p.strip() for p in arg.split(',') if p.strip()]
    for p in parts:
        if '=' not in p:
            continue
        k, v = p.split('=', 1)
        mapping[k.strip()] = v.strip()
    return mapping if mapping else None


def load_model_and_sampler(config: str, ckpt: str, device: torch.device, device_map: Optional[Dict[str, str]] = None):
    model = create_model(config).cpu()
    state_dict = load_state_dict(ckpt, location=str(device))
    model.load_state_dict(state_dict, strict=False)
    if device_map and hasattr(model, 'set_device_map'):
        model.set_device_map(device_map)
        primary = device_map.get('unet', str(device))
        device = torch.device(primary)
    model = model.to(device)
    model.eval()
    sampler = DDIMSampler(model)
    return model, sampler


@torch.no_grad()
def run_object_removal(
    config: str,
    ckpt: str,
    input_path: str,
    mask_path: str,
    output_path: str,
    prompt: str,
    a_prompt: str = "",
    n_prompt: str = "",
    control_path: Optional[str] = None,
    control_strength: float = 1.0,
    guess_mode: bool = False,
    steps: int = 50,
    scale: float = 9.0,
    eta: float = 0.0,
    seed: int = -1,
    device_str: str = "cuda",
    image_resolution: Optional[int] = None,
    # mask/inversion options
    use_inversion_start: bool = True,
    t_enc: Optional[int] = None,
    t_enc_ratio: Optional[float] = 0.5,
    mask_blur_sigma: float = 3.0,
    mask_dilate_iter: int = 1,
    # regional controls
    cfg_inside: float = 9.0,
    cfg_outside: float = 1.2,
    noise_mult_inside: float = 1.4,
    noise_mult_outside: float = 1.0,
    # repaint
    repaint_enabled: bool = False,
    repaint_jump: int = 5,
    repaint_every_k: int = 10,
    # memory controls
    use_fp16: bool = False,
    use_bf16: bool = False,
    channels_last: bool = False,
    tf32: bool = True,
    cudnn_benchmark: bool = True,
    use_compile: bool = False,
    low_vram: bool = False,
    # model parallel
    device_map_arg: Optional[str] = None,
):
    device = torch.device(device_str if torch.cuda.is_available() and device_str.startswith("cuda") else "cpu")

    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    if device.type == 'cuda':
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision('high' if tf32 else 'medium')
        except Exception:
            pass

    if (use_fp16 or use_bf16) and device.type == 'cuda':
        os.environ.setdefault("ATTN_PRECISION", "fp16")
    else:
        os.environ.setdefault("ATTN_PRECISION", "fp32")

    # Load model & sampler
    device_map = _parse_device_map_arg(device_map_arg)
    model, sampler = load_model_and_sampler(config, ckpt, device, device_map=device_map)
    unet_device = getattr(model, 'device', device)

    if channels_last:
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                model.model.diffusion_model.to(memory_format=torch.channels_last)
            if hasattr(model, 'control_model') and model.control_model is not None:
                model.control_model.to(memory_format=torch.channels_last)
        except Exception:
            pass

    if (use_fp16 or use_bf16) and torch.device(unet_device).type == 'cuda':
        target_dtype = torch.bfloat16 if use_bf16 else torch.float16
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                model.model.diffusion_model.to(dtype=target_dtype)
            if hasattr(model, 'control_model') and model.control_model is not None:
                model.control_model.to(dtype=target_dtype)
        except Exception:
            pass

    if use_compile:
        try:
            compile_fn = getattr(torch, 'compile', None)
            if callable(compile_fn):
                if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                    model.model.diffusion_model = compile_fn(model.model.diffusion_model, mode='max-autotune')
                if hasattr(model, 'control_model') and model.control_model is not None:
                    model.control_model = compile_fn(model.control_model, mode='max-autotune')
        except Exception:
            pass

    img_rgb = _read_image(input_path, target_size=image_resolution)
    h, w, _ = img_rgb.shape
    mask_inside = _read_mask(mask_path, (h, w))

    # Build soft masks
    inside_soft, keep_soft = _make_soft_masks(mask_inside, blur_sigma=mask_blur_sigma, dilate_iter=mask_dilate_iter)

    # To tensors
    vae_device = next(model.first_stage_model.parameters()).device if hasattr(model, 'first_stage_model') else unet_device
    x_image = _to_torch_image(img_rgb, vae_device)
    z0 = model.get_first_stage_encoding(model.encode_first_stage(x_image))  # (1,4,H/8,W/8)
    if z0.device != unet_device:
        z0 = z0.to(unet_device, non_blocking=True)
    latent_h, latent_w = z0.shape[-2:]

    keep_mask_soft_lat = _downsample_mask(keep_soft, (latent_h, latent_w))
    keep_mask_hard_lat = _downsample_mask(1.0 - mask_inside, (latent_h, latent_w))  # keep=1, edit=0
    edit_mask_inside_lat = _downsample_mask(mask_inside, (latent_h, latent_w))       # inside=1

    mask_soft_t = _to_torch_map(keep_mask_soft_lat, unet_device)
    mask_hard_t = _to_torch_map(keep_mask_hard_lat, unet_device)
    inside_mask_t = _to_torch_map(edit_mask_inside_lat, unet_device)

    if channels_last and torch.device(unet_device).type == 'cuda':
        try:
            z0 = z0.contiguous(memory_format=torch.channels_last)
            mask_soft_t = mask_soft_t.contiguous(memory_format=torch.channels_last)
            mask_hard_t = mask_hard_t.contiguous(memory_format=torch.channels_last)
            inside_mask_t = inside_mask_t.contiguous(memory_format=torch.channels_last)
        except Exception:
            pass

    control_t = None
    if control_path is not None and os.path.isfile(control_path):
        ctrl = cv2.imread(control_path, cv2.IMREAD_COLOR)
        if ctrl is None:
            raise FileNotFoundError(control_path)
        ctrl = cv2.cvtColor(ctrl, cv2.COLOR_BGR2RGB)
        ctrl = cv2.resize(ctrl, (w, h), interpolation=cv2.INTER_NEAREST)
        ctrl = (ctrl.astype(np.float32) / 255.0)
        control_t = torch.from_numpy(ctrl).float()
        if unet_device.type == 'cuda':
            try:
                control_t = control_t.pin_memory()
            except Exception:
                pass
        control_t = control_t.to(unet_device, non_blocking=True)
        control_t = einops.rearrange(control_t, 'h w c -> 1 c h w').contiguous()
        if channels_last and unet_device.type == 'cuda':
            try:
                control_t = control_t.contiguous(memory_format=torch.channels_last)
            except Exception:
                pass

    # Conditioning
    cond_prompt = prompt if not a_prompt else f"{prompt}, {a_prompt}"
    c_cross = model.get_learned_conditioning([cond_prompt])
    n_cross = model.get_learned_conditioning([n_prompt]) if n_prompt is not None else model.get_learned_conditioning([""])

    cond: Dict[str, list] = {"c_crossattn": [c_cross], "c_concat": [control_t] if control_t is not None else None}
    un_cond: Dict[str, list] = {
        "c_crossattn": [n_cross],
        "c_concat": None if guess_mode else ([control_t] if control_t is not None else None),
    }

    # Control strength schedule
    if control_t is not None:
        model.control_scales = [control_strength * (0.825 ** float(12 - i)) for i in range(13)]

    # Regional config dicts
    regional_cfg = {
        "mask": inside_mask_t,              # inside=1
        "scale_inside": float(cfg_inside),
        "scale_outside": float(cfg_outside),
    }
    regional_noise = {
        "mask": inside_mask_t,             # inside=1
        "mult_inside": float(noise_mult_inside),
        "mult_outside": float(noise_mult_outside),
    }
    repaint_cfg = {
        "enabled": bool(repaint_enabled),
        "jump": int(repaint_jump),
        "every_k": int(repaint_every_k),
    }

    # Sampler call
    # Low VRAM offloading during diffusion
    if low_vram and (device_map is None) and hasattr(model, 'low_vram_shift'):
        model.low_vram_shift(is_diffusing=True)
    try:
        if (use_fp16 or use_bf16) and torch.device(unet_device).type == 'cuda':
            autocast_ctx = torch.autocast('cuda', dtype=(torch.bfloat16 if use_bf16 else torch.float16))
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()
        with autocast_ctx:
            samples, _ = sampler.sample(
                S=int(steps),
                batch_size=1,
                shape=(4, latent_h, latent_w),
                conditioning=cond,
                eta=float(eta),
                mask=mask_hard_t,           # keep=1, edit=0
                x0=z0,
                temperature=1.0,
                verbose=False,
                unconditional_guidance_scale=float(scale),
                unconditional_conditioning=un_cond,
                # new options
                mask_soft=mask_soft_t,                      # soft keep-mask
                regional_cfg=regional_cfg,
                regional_noise=regional_noise,
                use_inversion_start=bool(use_inversion_start),
                t_enc=t_enc if t_enc is not None else None,
                t_enc_ratio=t_enc_ratio if (t_enc is None) else None,
                repaint_cfg=repaint_cfg,
            )
    finally:
        if low_vram and (device_map is None) and hasattr(model, 'low_vram_shift'):
            model.low_vram_shift(is_diffusing=False)

    # Decode and save
    # Ensure samples on VAE device for decoding
    if samples.device != vae_device:
        samples = samples.to(vae_device, non_blocking=True)
    x_dec = model.decode_first_stage(samples)
    x_np = x_dec.clamp(-1, 1)
    x_np = (einops.rearrange(x_np, '1 c h w -> h w c').cpu().numpy() + 1.0) * 0.5
    x_np = (x_np * 255.0).clip(0, 255).astype(np.uint8)
    x_bgr = cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, x_bgr)

    return output_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ControlNet object removal (train-free) with mask-aware DDIM and regional CFG/Noise")
    parser.add_argument('--config', type=str, required=True, help='Path to model yaml (e.g., ./models/cldm_v15.yaml)')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to ControlNet weights (e.g., ./models/control_sd15_canny.pth)')
    parser.add_argument('--input', type=str, required=True, help='Input RGB image path')
    parser.add_argument('--mask', type=str, required=True, help='Mask image path (white=remove region)')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--control', type=str, default=None, help='Optional control hint image path')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--a_prompt', type=str, default='', help='Additional positive prompt')
    parser.add_argument('--n_prompt', type=str, default='low quality, bad quality, artifacts', help='Negative prompt')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--scale', type=float, default=9.0, help='CFG scale')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--image_resolution', type=int, default=None, help='Resize longest side to this before inference')
    # memory controls
    parser.add_argument('--fp16', action='store_true', help='Use float16 for UNet/ControlNet and autocast')
    parser.add_argument('--bf16', action='store_true', help='Use bfloat16 for UNet/ControlNet and autocast')
    parser.add_argument('--channels_last', action='store_true', help='Use channels_last memory format for conv-heavy modules')
    parser.add_argument('--tf32', action='store_true', help='Enable TF32 matmul/cudnn for faster FP32 ops (Ampere+)')
    parser.add_argument('--no-tf32', action='store_true', help='Disable TF32')
    parser.add_argument('--cudnn_benchmark', action='store_true', help='Enable cuDNN benchmark for best conv algos')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile on UNet/ControlNet forward (PyTorch 2.x)')
    parser.add_argument('--low_vram', action='store_true', help='Offload VAE/Text to CPU during diffusion')
    # model parallel (comma-separated: unet=cuda:0,control=cuda:1,vae=cuda:2,clip=cuda:3)
    parser.add_argument('--device_map', type=str, default=None, help='Module-to-device mapping for model-parallel inference, e.g. "unet=cuda:0,control=cuda:1,vae=cuda:2,clip=cuda:3"')
    # mask + inversion
    parser.add_argument('--no_inversion', action='store_true', help='Disable DDIM inversion start')
    parser.add_argument('--t_enc', type=int, default=None, help='Inversion steps (overrides ratio)')
    parser.add_argument('--t_enc_ratio', type=float, default=0.5, help='Inversion ratio in [0,1]')
    parser.add_argument('--mask_blur_sigma', type=float, default=3.0)
    parser.add_argument('--mask_dilate_iter', type=int, default=1)
    # regional cfg + noise
    parser.add_argument('--cfg_inside', type=float, default=9.0)
    parser.add_argument('--cfg_outside', type=float, default=1.2)
    parser.add_argument('--noise_mult_inside', type=float, default=1.4)
    parser.add_argument('--noise_mult_outside', type=float, default=1.0)
    # repaint
    parser.add_argument('--repaint', action='store_true', help='Enable RePaint-style re-noise')
    parser.add_argument('--repaint_jump', type=int, default=5)
    parser.add_argument('--repaint_every_k', type=int, default=10)
    # control
    parser.add_argument('--control_strength', type=float, default=1.0)
    parser.add_argument('--guess_mode', action='store_true', help='Disable control in unconditional branch')
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    out = run_object_removal(
        config=args.config,
        ckpt=args.ckpt,
        input_path=args.input,
        mask_path=args.mask,
        output_path=args.output,
        prompt=args.prompt,
        a_prompt=args.a_prompt,
        n_prompt=args.n_prompt,
        control_path=args.control,
        control_strength=args.control_strength,
        guess_mode=args.guess_mode,
        steps=args.steps,
        scale=args.scale,
        eta=args.eta,
        seed=args.seed,
        device_str=args.device,
        image_resolution=args.image_resolution,
        use_inversion_start=(not args.no_inversion),
        t_enc=args.t_enc,
        t_enc_ratio=args.t_enc_ratio,
        mask_blur_sigma=args.mask_blur_sigma,
        mask_dilate_iter=args.mask_dilate_iter,
        cfg_inside=args.cfg_inside,
        cfg_outside=args.cfg_outside,
        noise_mult_inside=args.noise_mult_inside,
        noise_mult_outside=args.noise_mult_outside,
        repaint_enabled=args.repaint,
        repaint_jump=args.repaint_jump,
        repaint_every_k=args.repaint_every_k,
        use_fp16=args.fp16,
        use_bf16=args.bf16,
        channels_last=args.channels_last,
        tf32=(True if args.tf32 else (False if args.no_tf32 else True)),
        cudnn_benchmark=args.cudnn_benchmark,
        use_compile=args.compile,
        low_vram=args.low_vram,
        device_map_arg=args.device_map,
    )
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()


