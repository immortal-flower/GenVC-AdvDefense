"""
run_flf2v_experiment.py — FLF2V (First-Last-Frame) Chained-GOP Experiment on UVG
"""

import sys
import os
import re
import time
import json
import gc
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime
from accelerate import dispatch_model, infer_auto_device_map

# ================= 银弹：全局关闭梯度计算，显存占用暴跌 =================
torch.set_grad_enabled(False)
# ==============================================================================

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from sde_rf_wan.wan_flf2v_wrapper import WanFLF2VWrapper
from sde_rf_wan.turbo_pipeline import TurboDDCMWanPipeline
from sde_rf_wan.sde_convert import velocity_to_score, diffusion_coeff, sde_drift
from sde_rf_wan.ref_codec import compress_ref
from uvg_data import find_uvg_sequences as find_uvg_sequences_shared

# ==================================================================
# UVG Loading
# ==================================================================
def load_yuv420_frames(yuv_path, num_frames, start_frame=0):
    import cv2
    match = re.search(r'(\d+)x(\d+)', os.path.basename(yuv_path))
    if match:
        W, H = int(match.group(1)), int(match.group(2))
    else:
        W, H = 1280, 720
    frame_size = H * W * 3 // 2
    frames = []
    with open(yuv_path, 'rb') as f:
        f.seek(start_frame * frame_size)
        for _ in range(num_frames):
            raw = f.read(frame_size)
            if len(raw) < frame_size: break
            yuv = np.frombuffer(raw, dtype=np.uint8)
            y = yuv[:H * W].reshape(H, W)
            u = yuv[H * W:H * W + H * W // 4].reshape(H // 2, W // 2)
            v = yuv[H * W + H * W // 4:].reshape(H // 2, W // 2)
            u = cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
            yuv_img = np.stack([y, u, v], axis=-1)
            rgb = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)
            frames.append(Image.fromarray(rgb))
    return frames

def find_uvg_sequences(data_dir):
    return find_uvg_sequences_shared(data_dir)

def resize_frames(frames, target_w, target_h):
    return [f.resize((target_w, target_h), Image.LANCZOS) for f in frames]

def compress_boundary_frame(image, ref_codec="compressai", ref_quality=4):
    decoded, _, nbytes = compress_ref(image, codec=ref_codec, quality=ref_quality)
    return decoded, nbytes

# ==================================================================
# Metrics
# ==================================================================
def frames_to_tensor(frames):
    return torch.stack([torch.from_numpy(np.array(f).astype(np.float32) / 255.0).permute(2, 0, 1) for f in frames])

def compute_psnr(orig, recon):
    mse = ((orig - recon) ** 2).mean(dim=[1, 2, 3])
    psnr = -10.0 * torch.log10(mse + 1e-10)
    return psnr.mean().item(), psnr

def compute_msssim(orig, recon):
    try:
        from pytorch_msssim import ms_ssim
        vals = []
        for i in range(0, orig.shape[0], 4):
            v = ms_ssim(orig[i:i+4], recon[i:i+4], data_range=1.0, size_average=False)
            vals.extend(v.cpu().tolist())
        return sum(vals) / len(vals)
    except ImportError:
        return None

def compute_lpips(orig, recon, device="cuda:1"):
    import lpips
    loss_fn = lpips.LPIPS(net='alex').to(device)
    orig_lp = (2.0 * orig - 1.0).to(device)
    recon_lp = (2.0 * recon - 1.0).to(device)
    vals = []
    for i in range(0, orig_lp.shape[0], 4):
        with torch.no_grad():
            d = loss_fn(orig_lp[i:i+4], recon_lp[i:i+4])
        vals.extend(d.flatten().cpu().tolist())
    del loss_fn
    torch.cuda.empty_cache()
    return sum(vals) / len(vals)

def save_video_mp4(frames, path, fps=16):
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec='libx264', quality=8)
    for f in frames: writer.append_data(np.array(f))
    writer.close()

# ==================================================================
# FLF2V encode / decode helpers
# ==================================================================
def flf2v_encode(pipe, model, gop_frames, flf2v_cond, height, width):
    embeds = model.encode_prompt("")
    x0_true = model.encode_video(gop_frames, height, width)
    
    model_fn = pipe._model_fn(embeds, flf2v_cond)
    gen = torch.Generator(device="cpu").manual_seed(pipe.seed)
    x_t = torch.randn(1, *pipe.latent_shape, generator=gen).to(pipe.device)

    step_data = []
    sde_idx = 0

    for i in range(pipe.num_steps):
        t_curr = pipe.timesteps[i].item()
        t_next = pipe.timesteps[i + 1].item()
        delta_t = t_curr - t_next
        u_t = model_fn(x_t, t_curr)

        if t_next < 1e-6:
            x_t = x_t - u_t * delta_t
            break
        if i >= pipe.num_sde_steps:
            x_t = x_t - u_t * delta_t
            continue

        x0_hat = x_t - t_curr * u_t
        residual = (x0_true - x0_hat).squeeze(0)
        score = velocity_to_score(u_t, x_t, t_curr)
        g_t = diffusion_coeff(t_curr, pipe.g_scale)
        f_t = sde_drift(u_t, score, g_t)
        noise_coeff = g_t * (delta_t ** 0.5)

        frame_entries = []
        noise_frames = []
        for f in range(pipe.num_latent_frames):
            r_f = residual[:, f, :, :]
            M_f = pipe._get_M_for_frame(f)
            idx, sgn, z_f = pipe.codebook.select_atoms(r_f, sde_idx, f, M_override=M_f)
            frame_entries.append((idx, sgn))
            noise_frames.append(z_f)

        step_data.append(frame_entries)
        noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0)
        x_t = x_t - f_t * delta_t + noise_coeff * noise_3d
        sde_idx += 1

        if (i + 1) % 5 == 0 or i == 0:
            mse = ((x0_true - x0_hat) ** 2).mean().item()
            print(f"    Encode step {i+1}/{pipe.num_steps}: residual_MSE={mse:.4f}, noise_coeff={noise_coeff:.4f}")

    return step_data, x0_true

def flf2v_decode(pipe, model, step_data, flf2v_cond):
    embeds = model.encode_prompt("")
    model_fn = pipe._model_fn(embeds, flf2v_cond)
    gen = torch.Generator(device="cpu").manual_seed(pipe.seed)
    x_t = torch.randn(1, *pipe.latent_shape, generator=gen).to(pipe.device)
    sde_idx = 0

    for i in range(pipe.num_steps):
        t_curr = pipe.timesteps[i].item()
        t_next = pipe.timesteps[i + 1].item()
        delta_t = t_curr - t_next
        u_t = model_fn(x_t, t_curr)

        if t_next < 1e-6:
            x_t = x_t - u_t * delta_t
            break
        if i >= pipe.num_sde_steps:
            x_t = x_t - u_t * delta_t
            continue

        score = velocity_to_score(u_t, x_t, t_curr)
        g_t = diffusion_coeff(t_curr, pipe.g_scale)
        f_t = sde_drift(u_t, score, g_t)
        noise_coeff = g_t * (delta_t ** 0.5)

        noise_frames = []
        for f in range(pipe.num_latent_frames):
            idx, sgn = step_data[sde_idx][f]
            z_f = pipe.codebook.reconstruct(idx, sgn, sde_idx, f)
            noise_frames.append(z_f)

        noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0)
        x_t = x_t - f_t * delta_t + noise_coeff * noise_3d
        sde_idx += 1
        if (i + 1) % 5 == 0:
            print(f"    Decode step {i+1}/{pipe.num_steps}")

    # ================= 终极显存过桥清洗 =================
    x_t_cpu = x_t.cpu() 
    del x_t, model_fn, embeds, noise_3d, noise_frames
    gc.collect()
    torch.cuda.empty_cache() 
    
    frames_recon = model.decode_latent(x_t_cpu)
    return frames_recon

# ==================================================================
# Main
# ==================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="FLF2V Chained-GOP UVG Experiment")
    parser.add_argument("--data_dir", default=os.path.join(_project_root, "data", "uvg"))
    parser.add_argument("--wan_ckpt", default="./Wan2.1-FLF2V-14B-720P")
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--num_frames_per_gop", type=int, default=33)
    parser.add_argument("--num_gops", type=int, default=3)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--K", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ddim_tail", type=int, default=3)
    parser.add_argument("--g_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref_codec", default="compressai", choices=["compressai", "webp", "gt"])
    parser.add_argument("--ref_quality", type=int, default=4)
    parser.add_argument("--flow_shift", type=float, default=None)
    parser.add_argument("--sequences", nargs="*", default=None)
    args = parser.parse_args()

    if args.flow_shift is None: args.flow_shift = 3.0 if args.height <= 480 else 5.0

    HEIGHT, WIDTH = args.height, args.width
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    FPG = args.num_frames_per_gop
    frames_per_gop_excl_first = FPG - 1 

    all_seqs = find_uvg_sequences(args.data_dir)
    if args.sequences: all_seqs = [(n, p) for n, p in all_seqs if n in args.sequences]
    if not all_seqs: sys.exit(1)

    # ================================================================
    # 核心模型加载 & 极致 22:9 偏载架构
    # ================================================================
    model = WanFLF2VWrapper(args.wan_ckpt, config_name="flf2v-14B", flow_shift=args.flow_shift)
    print("\n[GVCC] Loading 14B model safely to CPU first...")
    model.load("cpu", torch.bfloat16)
    
    print("[GVCC] Activating Hugging Face Accelerate for Dual RTX 5090s...")
    
    # 【致胜微操】：22:9 的黄金偏载！
    # GPU 0 吃下绝大部分参数 (22GB)，留 10GB 做推理想象空间。
    # GPU 1 只带 9GB 碎片，足足空出 23GB 的海量真空区，专治 VAE 洪峰！
    max_mem = {0: "22GiB", 1: "9GiB"}  
    device_map = infer_auto_device_map(model.model, max_memory=max_mem)
    
    clean_map = {}
    for key, value in device_map.items():
        match = re.search(r'(^.*?blocks\.\d+)', key)
        if match:
            block_key = match.group(1)
            if block_key not in clean_map:
                clean_map[block_key] = value
        else:
            clean_map[key] = 0
            
    model.model = dispatch_model(model.model, device_map=clean_map)
    model.device = torch.device("cuda:0")

    # ================= 动态拦截器 (全部打向真空区 GPU 1) =================
    print("[GVCC] Injecting Smart Memory Interceptors (Targeting Vacuum Zone GPU 1)...")
    
    if hasattr(model, 'clip') and model.clip is not None:
        original_clip_visual = model.clip.visual
        def patched_clip_visual(videos, *args, **kwargs):
            with torch.no_grad():
                c_mod = model.clip.model if hasattr(model.clip, 'model') else model.clip
                c_mod.to("cuda:1") # 去 GPU 1
                v_gpu = [v.to("cuda:1") for v in videos] if isinstance(videos, list) else videos.to("cuda:1")
                res = original_clip_visual(v_gpu, *args, **kwargs)
                c_mod.to("cpu")
                torch.cuda.empty_cache()
                if hasattr(res, 'to'): return res.to("cuda:0")
                elif isinstance(res, list): return [r.to("cuda:0") for r in res]
                return res
        model.clip.visual = patched_clip_visual

    original_encode_prompt = model.encode_prompt
    def patched_encode_prompt(prompt, *args, **kwargs):
        with torch.no_grad():
            torch.cuda.empty_cache()
            prev_dev = model.device
            model.device = torch.device("cuda:1") # 欺骗系统在 GPU1 生成
            if hasattr(model, 'text_encoder') and model.text_encoder is not None:
                t_mod = model.text_encoder.model if hasattr(model.text_encoder, 'model') else model.text_encoder
                t_mod.to("cuda:1")
            
            res = original_encode_prompt(prompt, *args, **kwargs)
            
            if hasattr(model, 'text_encoder') and model.text_encoder is not None:
                t_mod = model.text_encoder.model if hasattr(model.text_encoder, 'model') else model.text_encoder
                t_mod.to("cpu")
                
            model.device = prev_dev
            torch.cuda.empty_cache()
            # 搬回 GPU 0 交给大模型
            if isinstance(res, dict): return {k: (v.to("cuda:0") if hasattr(v, 'to') else v) for k, v in res.items()}
            elif hasattr(res, 'to'): return res.to("cuda:0")
            elif isinstance(res, list): return [r.to("cuda:0") if hasattr(r,'to') else r for r in res]
            elif isinstance(res, tuple): return tuple(r.to("cuda:0") if hasattr(r,'to') else r for r in res)
            return res
    model.encode_prompt = patched_encode_prompt

    if hasattr(model, 'vae') and model.vae is not None:
        original_vae_encode = model.vae.encode
        def patched_vae_encode(videos, *args, **kwargs):
            with torch.no_grad():
                torch.cuda.empty_cache() 
                v_mod = model.vae.model if hasattr(model.vae, 'model') else model.vae
                v_mod.to("cuda:1") # 路由到 GPU 1 宽阔跑道
                v_gpu = [v.to("cuda:1") for v in videos] if isinstance(videos, list) else videos.to("cuda:1")
                res = original_vae_encode(v_gpu, *args, **kwargs)
                v_mod.to("cpu")
                torch.cuda.empty_cache()
                if isinstance(res, list): return [r.to("cuda:0") for r in res]
                elif isinstance(res, tuple): return tuple(r.to("cuda:0") if hasattr(r, 'to') else r for r in res)
                else: return res.to("cuda:0")
        model.vae.encode = patched_vae_encode
        
        original_vae_decode = model.vae.decode
        def patched_vae_decode(zs, *args, **kwargs):
            with torch.no_grad():
                torch.cuda.empty_cache() 
                v_mod = model.vae.model if hasattr(model.vae, 'model') else model.vae
                v_mod.to("cuda:1") # 路由到 GPU 1 宽阔跑道
                z_gpu = [z.to("cuda:1") for z in zs] if isinstance(zs, list) else zs.to("cuda:1")
                res = original_vae_decode(z_gpu, *args, **kwargs)
                v_mod.to("cpu")
                torch.cuda.empty_cache()
                if isinstance(res, list): return [r.to("cuda:0") for r in res]
                elif isinstance(res, tuple): return tuple(r.to("cuda:0") if hasattr(r, 'to') else r for r in res)
                else: return res.to("cuda:0")
        model.vae.decode = patched_vae_decode

    model.model.cpu = lambda *args, **kwargs: model.model
    model.model.to = lambda *args, **kwargs: model.model
    print("[GVCC] All systems green. Asymmetric balancing activated (22:9 ratio).")
    # ====================================================

    all_seq_results = []
    for seq_idx, (seq_name, yuv_path) in enumerate(all_seqs):
        print(f"\n{'='*70}\n  [{seq_idx+1}/{len(all_seqs)}] Sequence: {seq_name}\n{'='*70}")
        seq_dir = out / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        total_unique_frames = frames_per_gop_excl_first * args.num_gops + 1 if args.num_gops > 0 else 999999
        raw_frames = load_yuv420_frames(yuv_path, total_unique_frames, start_frame=0)
        actual_gops = max(1, (len(raw_frames) - 1) // frames_per_gop_excl_first) if (args.num_gops <= 0 or len(raw_frames) < total_unique_frames) else args.num_gops
        actual_total = frames_per_gop_excl_first * actual_gops + 1
        raw_frames = raw_frames[:actual_total]
        frames_resized = resize_frames(raw_frames, WIDTH, HEIGHT)
        del raw_frames

        gops_gt = []
        for g in range(actual_gops):
            start = g * frames_per_gop_excl_first
            end = start + FPG
            gops_gt.append(frames_resized[start:end])

        save_video_mp4(frames_resized, seq_dir / "original.mp4")

        boundary_indices = [min(g * frames_per_gop_excl_first, len(frames_resized) - 1) for g in range(actual_gops + 1)]
        boundary_compressed = {}
        total_boundary_bytes = 0

        for idx in boundary_indices:
            gt_frame = frames_resized[idx]
            if args.ref_codec == "gt":
                boundary_compressed[idx] = (gt_frame, 0)
            else:
                decoded, nbytes = compress_boundary_frame(gt_frame, ref_codec=args.ref_codec, ref_quality=args.ref_quality)
                boundary_compressed[idx] = (decoded, nbytes)
                total_boundary_bytes += nbytes

        gop_results = []
        all_recon_frames = []

        for g in range(actual_gops):
            print(f"\n  --- {seq_name} GOP {g}/{actual_gops-1}  ({datetime.now().strftime('%H:%M:%S')}) ---")
            gop_frames = gops_gt[g]
            first_idx, last_idx = g * frames_per_gop_excl_first, (g + 1) * frames_per_gop_excl_first
            first_decoded, first_bytes = boundary_compressed[first_idx]
            last_decoded, last_bytes = boundary_compressed[last_idx]

            gop_first_bytes = 0 if g > 0 else first_bytes
            pipe = TurboDDCMWanPipeline(
                model, K=args.K, M=args.M, num_steps=args.steps, num_ddim_tail=args.ddim_tail,
                guidance_scale=1.0, g_scale=args.g_scale, num_frames=FPG,
                height=HEIGHT, width=WIDTH, seed=args.seed,
            )
            
            flf2v_cond = model.encode_first_last_frames(first_decoded, last_decoded, FPG, HEIGHT, WIDTH)
            
            print(f"  Encoding...")
            t0 = time.time()
            step_data, x0_true = flf2v_encode(pipe, model, gop_frames, flf2v_cond, HEIGHT, WIDTH)
            t_enc = time.time() - t0

            # ===== 终极清场 ======
            del flf2v_cond, x0_true
            gc.collect()
            torch.cuda.empty_cache()

            print(f"  Decoding...")
            t0 = time.time()
            flf2v_cond_dec = model.encode_first_last_frames(first_decoded, last_decoded, FPG, HEIGHT, WIDTH)
            frames_recon = flf2v_decode(pipe, model, step_data, flf2v_cond_dec)
            t_dec = time.time() - t0

            gop_dir = seq_dir / f"gop{g}"
            gop_dir.mkdir(parents=True, exist_ok=True)
            save_video_mp4(frames_recon, gop_dir / "reconstructed.mp4")
            pipe.save_compressed(step_data, str(gop_dir / "codebook.tdcm"))

            if g == 0: all_recon_frames.extend(frames_recon)
            else: all_recon_frames.extend(frames_recon[1:])

            n = min(len(gop_frames), len(frames_recon))
            t_gt, t_rec = frames_to_tensor(gop_frames[:n]), frames_to_tensor(frames_recon[:n])

            mean_psnr, per_frame_psnr = compute_psnr(t_gt, t_rec)
            mean_msssim = compute_msssim(t_gt, t_rec)
            mean_lpips = compute_lpips(t_gt, t_rec, device="cuda:1")

            T_sde, F_lat = pipe.num_sde_steps, pipe.num_latent_frames
            codebook_bytes = (T_sde * F_lat * pipe.codebook.bits_per_frame_step) // 8
            gop_boundary_bytes = gop_first_bytes + last_bytes
            gop_total_bytes = codebook_bytes + gop_boundary_bytes
            bpp = (gop_total_bytes * 8) / (FPG * HEIGHT * WIDTH)
            
            result = {
                "sequence": seq_name, "gop": g, "PSNR_dB": round(mean_psnr, 2),
                "LPIPS": round(mean_lpips, 4), "BPP": round(bpp, 6), "gop_total_bytes": gop_total_bytes,
            }
            gop_results.append(result)
            with open(gop_dir / "metrics.json", "w") as mf: json.dump(result, mf, indent=2)

            print(f"    PSNR={mean_psnr:.2f} dB, LPIPS={mean_lpips:.4f}, BPP={bpp:.6f}")
            del pipe, frames_recon, t_gt, t_rec, step_data, flf2v_cond_dec
            gc.collect()
            torch.cuda.empty_cache()

        save_video_mp4(all_recon_frames, seq_dir / "reconstructed_full.mp4")
        all_seq_results.append({"sequence": seq_name, "gop_results": gop_results})
        del frames_resized, all_recon_frames
        gc.collect()

    print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
    for sr in all_seq_results:
        for r in sr["gop_results"]:
            print(f"Seq: {r['sequence']} | GOP: {r['gop']} | PSNR: {r['PSNR_dB']} | LPIPS: {r['LPIPS']}")

if __name__ == "__main__":
    main()