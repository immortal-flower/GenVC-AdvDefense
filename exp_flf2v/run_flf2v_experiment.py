"""在 UVG 数据上运行 GVCC 的 Wan FLF2V（首尾帧条件）实验。

建议按 main() 的执行顺序阅读：读视频 → 攻击/防御预处理 → 切 GOP →
压缩首尾帧 → VAE 编码与码本选择 → 重放解码 → 计算指标与保存结果。
本文件负责组织一次实验；VAE/DiT 接口在 wan_flf2v_wrapper.py，
高斯码本的生成与 Top-M 选择在 turbo_codebook.py。

找“数据入口”时注意：本脚本没有 PyTorch Dataset/DataLoader。
main() 先调用 find_uvg_sequences() 查找 .yuv 路径（还没有读取像素），
之后在序列循环里调用 load_yuv420_frames()，这才真正打开文件读入视频。
可以先跳过中间较长的双 GPU 内存分配代码，顺着这两处函数调用往下看。
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

# 当前是推理/指标脚本：关闭 autograd 可避免保存 14B 模型的反向传播激活。
# 若以后做基于输入梯度的攻击，需要单独建立可微路径；只设置输入
# requires_grad=True 不能越过这里和 wrapper 内部的 no_grad。
torch.set_grad_enabled(False)
# ==============================================================================

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 允许从项目根目录导入 sde_rf_wan、uvg_data 等模块。
sys.path.insert(0, _project_root)
from sde_rf_wan.wan_flf2v_wrapper import WanFLF2VWrapper
from sde_rf_wan.turbo_pipeline import TurboDDCMWanPipeline
from sde_rf_wan.sde_convert import velocity_to_score, diffusion_coeff, sde_drift
from sde_rf_wan.ref_codec import compress_ref
from uvg_data import find_uvg_sequences as find_uvg_sequences_shared

# ==================================================================
# UVG 原始 YUV420 读取与参考帧压缩
# ==================================================================
def load_yuv420_frames(yuv_path, num_frames, start_frame=0):
    # 参数：yuv_path 是单个文件路径；num_frames 是最多读取多少帧；
    # start_frame=0 表示从第 0 帧开始。返回 list[PIL.Image]，不是 Tensor。
    import cv2
    # 文件名若含 1280x720 等分辨率则采用它，否则按本实验默认的 720p 读取。
    # 原始 .yuv 通常没有能自动说明分辨率的文件头；若实际文件不是 720p，
    # 而文件名又没有宽×高，下面的默认尺寸会把字节流按错误帧长切开。
    match = re.search(r'(\d+)x(\d+)', os.path.basename(yuv_path))
    if match:
        W, H = int(match.group(1)), int(match.group(2))
    else:
        W, H = 1280, 720
    # YUV420 每帧：Y 为 H×W，U/V 各为 H/2×W/2，总计 1.5HW 字节。
    frame_size = H * W * 3 // 2
    frames = []
    # 这里才真正打开磁盘文件；'rb' 表示按二进制读取原始字节。
    # with 代码块结束后文件自动关闭，f 是当前打开文件的句柄。
    with open(yuv_path, 'rb') as f:
        # seek(n) 把读取位置移到第 n 个字节；可跳过前面的 start_frame 帧。
        f.seek(start_frame * frame_size)
        # _ 表示循环变量本身不用；若文件提前结束就退出循环。
        for _ in range(num_frames):
            raw = f.read(frame_size)
            if len(raw) < frame_size: break
            yuv = np.frombuffer(raw, dtype=np.uint8)
            # 平面格式的 Y、U、V 分开存放；U/V 先放大至 H×W，再转 RGB。
            # [:H*W] 等是数组切片；reshape 只改数组形状，不增加像素。
            y = yuv[:H * W].reshape(H, W)
            u = yuv[H * W:H * W + H * W // 4].reshape(H // 2, W // 2)
            v = yuv[H * W + H * W // 4:].reshape(H // 2, W // 2)
            u = cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
            yuv_img = np.stack([y, u, v], axis=-1)
            rgb = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)
            # 返回值是 PIL.Image 列表，每张图的大小为 W×H、3 个 RGB 通道。
            frames.append(Image.fromarray(rgb))
    return frames

def find_uvg_sequences(data_dir):
    # 这一层只转交给 uvg_data.py：查找 .yuv 文件并返回
    # [(序列名, 文件路径), ...]；不会把视频画面读进内存。
    return find_uvg_sequences_shared(data_dir)

def resize_frames(frames, target_w, target_h):
    # 注意这里仍是像素域 RGB；90×160 的潜空间尺寸要到 VAE 编码后才出现。
    # [表达式 for f in frames] 是列表推导式：对每一张 PIL 图片做相同缩放。
    return [f.resize((target_w, target_h), Image.LANCZOS) for f in frames]

def compress_boundary_frame(image, ref_codec="compressai", ref_quality=4):
    # FLF2V 的首、尾参考帧需要传给解码端；这里返回解压后的共同条件
    # 和该参考帧的压缩字节数，避免编解码两侧用到不同版本的边界帧。
    decoded, _, nbytes = compress_ref(image, codec=ref_codec, quality=ref_quality)
    return decoded, nbytes

# ==================================================================
# 指标：输入为整段 GOP 的 [帧数, 3, H, W]、取值范围 [0, 1]
# ==================================================================
def frames_to_tensor(frames):
    # 每张 PIL 图片先变 NumPy [H,W,3]，再归一化到 [0,1]；
    # permute(2,0,1) 调整为 PyTorch 常用的 [3,H,W]；stack 组成 [F,3,H,W]。
    return torch.stack([torch.from_numpy(np.array(f).astype(np.float32) / 255.0).permute(2, 0, 1) for f in frames])

def compute_psnr(orig, recon):
    # 先对每一帧的 C/H/W 求 MSE，再把每帧 PSNR（dB）取算术平均。
    # 因此 mean(逐帧 PSNR) 不等于先平均所有像素 MSE 后再算 PSNR。
    # orig/recon 都是 [F,3,H,W]；dim=[1,2,3] 表示只压掉通道和空间维，留下 F。
    mse = ((orig - recon) ** 2).mean(dim=[1, 2, 3])
    psnr = -10.0 * torch.log10(mse + 1e-10)
    return psnr.mean().item(), psnr

def compute_msssim(orig, recon):
    # 分批减少指标计算时的显存占用；缺少依赖时返回 None。
    # orig[i:i+4] 是左闭右开的切片，一次取最多 4 帧。
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
    # LPIPS-Alex 接受 [-1, 1] 图像，逐帧计算后对整个 GOP 取平均。
    # device="cuda:1" 仅指定此项指标使用 GPU 1，不代表主 DiT 在 GPU 1。
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
    # MP4 仅用于查看结果；指标在保存前的内存帧上计算，MP4 有额外有损误差。
    # frames 是 PIL 图片列表；写入视频时再逐帧转回 NumPy 数组。
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec='libx264', quality=8)
    for f in frames: writer.append_data(np.array(f))
    writer.close()

from attacks import ATTACK_CHOICES, apply_attack
from defenses import DEFENSE_CHOICES, apply_defense
# ==================================================================
# FLF2V 编解码核心：两端共享模型、随机种子、首尾帧条件和时间步。
# 编码端额外拥有原视频，可计算 x0_true 并挑选码本索引；解码端只有索引。
# ==================================================================
def flf2v_encode(pipe, model, gop_frames, flf2v_cond, height, width):
    # gop_frames：一个 GOP 的 RGB 帧列表，默认长度 33，每张 1280×720。
    # flf2v_cond：由边界首尾帧产生的条件字典，不包含完整的 33 帧原视频。
    # pipe 维护时间步、共享码本及噪声种子；model 是已经加载好的 Wan 包装器。
    # 本实验使用空文本提示词。它在编解码两侧各自编码，并未传输特征张量。
    embeds = model.encode_prompt("")
    # VAE 把 RGB GOP 编成目标潜变量 x0_true。
    # 33×720×1280 帧对应 [1,16,9,90,160]：
    # batch=1，潜通道=16，潜在时间帧=(33-1)/4+1=9，空间=720/8×1280/8。
    # 原视频只在编码端通过这个 x0_true 计算残差；解码端没有它。
    x0_true = model.encode_video(gop_frames, height, width)
    
    # model_fn(x_t, t) 以当前潜变量、时间和首尾帧条件预测速度 u_t。
    model_fn = pipe._model_fn(embeds, flf2v_cond)
    # 固定种子的初始高斯噪声是双方的共同起点，本身不携带本视频的内容。
    gen = torch.Generator(device="cpu").manual_seed(pipe.seed)
    # *pipe.latent_shape 将 (16,9,90,160) 作为四个参数展开；
    # 前面的 1 是 batch 维，得到与 x0_true 同形状的 x_t。
    x_t = torch.randn(1, *pipe.latent_shape, generator=gen).to(pipe.device)

    # step_data 的结构为 [SDE 步][潜在时间帧] = (M 个索引, M 个符号)。
    step_data = []
    sde_idx = 0

    for i in range(pipe.num_steps):
        # 时间从噪声侧走向干净侧；delta_t 是本次更新的时间间隔。
        # 例如 steps=20、ddim_tail=3：循环最多 20 次，其中前 17 次选码本。
        t_curr = pipe.timesteps[i].item()
        t_next = pipe.timesteps[i + 1].item()
        delta_t = t_curr - t_next
        # u_t 与 x_t 形状相同；模型看的是当前 x_t、时间和条件，而非完整原视频。
        u_t = model_fn(x_t, t_curr)

        if t_next < 1e-6:
            # 最后到 t=0 的收尾更新是确定性的，不检索码本。
            # break：直接结束整个 for 循环；解码端也执行同样的末步。
            x_t = x_t - u_t * delta_t
            break
        if i >= pipe.num_sde_steps:
            # DDIM/ODE 尾部同样不选原子，因此不增加 step_data。
            # continue：只结束本轮循环，进入下一个时间步。
            x_t = x_t - u_t * delta_t
            continue

        # 由当前位置与速度估计最终干净潜变量，再与真实 x0 比较。
        x0_hat = x_t - t_curr * u_t
        # squeeze(0) 去掉 batch=1 维：[1,16,9,90,160] → [16,9,90,160]。
        residual = (x0_true - x0_hat).squeeze(0)
        # f_t 是确定性漂移，noise_coeff 决定码本合成噪声的注入强度。
        score = velocity_to_score(u_t, x_t, t_curr)
        g_t = diffusion_coeff(t_curr, pipe.g_scale)
        f_t = sde_drift(u_t, score, g_t)
        noise_coeff = g_t * (delta_t ** 0.5)

        frame_entries = []
        noise_frames = []
        # 注意这里遍历的是 9 个 VAE 潜在时间帧，不是 33 个 RGB 视频帧。
        for f in range(pipe.num_latent_frames):
            # 单个 r_f 的形状是 [16, 90, 160]；从 K 个共享高斯原子中
            # 选 |内积| 最大的 M 个，并记录索引 idx 和正负方向 sgn。
            # residual[:, f, :, :]：第 1 维的 ':' 表示取全部 16 个通道；
            # f 只取一个潜在时间帧；后两个 ':' 取完整 90×160 空间。
            r_f = residual[:, f, :, :]
            M_f = pipe._get_M_for_frame(f)
            # 右边返回三个值，左边一次“解包”到 idx、sgn、z_f：
            # idx/sgn 是待传输信息；z_f 是由所选原子合成的本帧纠偏噪声。
            idx, sgn, z_f = pipe.codebook.select_atoms(r_f, sde_idx, f, M_override=M_f)
            frame_entries.append((idx, sgn))
            noise_frames.append(z_f)

        step_data.append(frame_entries)
        # 将 9 个合成噪声叠回 [1, 16, 9, 90, 160]，修正下一时刻轨迹。
        # stack(..., dim=1) 在通道后插入时间维；unsqueeze(0) 再插入 batch 维。
        noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0)
        # SDE 更新 = 模型给出的漂移运动 + 码本选出的随机纠偏方向。
        x_t = x_t - f_t * delta_t + noise_coeff * noise_3d
        sde_idx += 1

        if (i + 1) % 5 == 0 or i == 0:
            mse = ((x0_true - x0_hat) ** 2).mean().item()
            print(f"    Encode step {i+1}/{pipe.num_steps}: residual_MSE={mse:.4f}, noise_coeff={noise_coeff:.4f}")

    # 返回两个对象给 main()：码本索引/符号，以及仅编码端使用的真值潜变量。
    return step_data, x0_true

def flf2v_decode(pipe, model, step_data, flf2v_cond):
    # 解码端不调用 encode_video(gop_frames)，也不计算真实残差。
    # step_data 是编码端得到的 [SDE步][潜在时间帧] 列表；每格是一对 (idx,sgn)。
    # 注意本实验函数直接复用内存里的 step_data，没有从 .tdcm 文件再读取一次。
    embeds = model.encode_prompt("")
    model_fn = pipe._model_fn(embeds, flf2v_cond)
    gen = torch.Generator(device="cpu").manual_seed(pipe.seed)
    x_t = torch.randn(1, *pipe.latent_shape, generator=gen).to(pipe.device)
    # 相同种子 + 相同潜变量形状，复现编码端的初始 x_t。
    sde_idx = 0

    for i in range(pipe.num_steps):
        # 时间表和速度预测与编码端对应；末尾确定性步骤不用读取码本数据。
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
            # 利用收到的索引/符号和共享种子重新生成原子与合成噪声。
            # step_data[sde_idx][f] 先选第几次 SDE，再选第几个潜在时间帧。
            idx, sgn = step_data[sde_idx][f]
            z_f = pipe.codebook.reconstruct(idx, sgn, sde_idx, f)
            noise_frames.append(z_f)

        noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0)
        x_t = x_t - f_t * delta_t + noise_coeff * noise_3d
        sde_idx += 1
        if (i + 1) % 5 == 0:
            print(f"    Decode step {i+1}/{pipe.num_steps}")

    # 采样已结束：把潜变量转到 CPU，为随后占显存较多的 VAE 解码腾空间。
    # 这里的 x_t 是重建出的潜变量，不是 RGB 图片。
    x_t_cpu = x_t.cpu() 
    del x_t, model_fn, embeds, noise_3d, noise_frames
    gc.collect()
    torch.cuda.empty_cache() 
    
    # VAE 解码回 RGB 帧列表；最终视频仍是 33 帧、720×1280。
    frames_recon = model.decode_latent(x_t_cpu)
    return frames_recon

# ==================================================================
# Main
# ==================================================================
def main():
    # Python 从文件末尾的 if __name__ == "__main__" 进入这里。
    # 建议先读参数 → 找文件 → 暂跳双 GPU 配置 → 从“真正读数据”处继续。
    import argparse
    parser = argparse.ArgumentParser(description="FLF2V Chained-GOP UVG Experiment")
    # argparse 负责把命令行选项保存到 args，例如：
    # --data_dir "D:\...\Jockey_720p.yuv" 会成为 args.data_dir 字符串。
    # 路径和视频范围：一个 GOP 默认 33 帧；相邻 GOP 共用首尾边界帧。
    parser.add_argument("--data_dir", default=os.path.join(_project_root, "data", "uvg"))
    # wan_ckpt 是模型权重目录；output_dir 是实验输出目录，不是输入视频目录。
    parser.add_argument("--wan_ckpt", default="./Wan2.1-FLF2V-14B-720P")
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--num_frames_per_gop", type=int, default=33)
    parser.add_argument("--num_gops", type=int, default=3)
    # type=int/float 表示解析后得到数值；不传选项时才使用 default。
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    # GVCC 码本参数：每个潜在时间帧、每个 SDE 步从 K 个候选原子中选 M 个。
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--K", type=int, default=16384)
    # 前 steps - ddim_tail 步注入码本噪声；最后 ddim_tail 步确定性收尾。
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ddim_tail", type=int, default=3)
    parser.add_argument("--g_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref_codec", default="compressai", choices=["compressai", "webp", "gt"])
    parser.add_argument("--ref_quality", type=int, default=4)
    parser.add_argument("--flow_shift", type=float, default=None)
    # nargs="*"：可跟零个或多个序列名，如 --sequences Jockey Beauty。
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--run_name", default=None, help="Optional subdirectory name under output_dir")
    # 攻击和防御都发生在输入帧上，不直接改 Wan / VAE 权重或码本索引。
    parser.add_argument("--attack", default="none", choices=ATTACK_CHOICES)
    parser.add_argument("--defense", default="none", choices=DEFENSE_CHOICES)
    parser.add_argument("--epsilon", type=float, default=4.0, help="Attack budget; values > 1 are interpreted as pixel levels out of 255")
    parser.add_argument("--attack_steps", type=int, default=8)
    parser.add_argument("--attack_alpha", type=float, default=0.0, help="Attack step size; 0 uses epsilon / attack_steps")
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--median_size", type=int, default=3)
    # 之前只是在“声明”可接受哪些选项；这一行才解析实际命令行。
    args = parser.parse_args()

    # Wan 的时间步偏移随分辨率使用不同默认值；显式传参可覆盖。
    if args.flow_shift is None: args.flow_shift = 3.0 if args.height <= 480 else 5.0

    # 左边两个变量同时接收右边两个值，称为解包赋值。
    HEIGHT, WIDTH = args.height, args.width
    # 不同实验写入不同目录，避免直接覆盖 clean / attack / defense 的结果。
    # Path 是路径对象；out / args.run_name 表示拼接一个子目录，不是数值除法。
    out = Path(args.output_dir)
    if args.run_name:
        out = out / args.run_name
    elif args.attack != "none" or args.defense != "none":
        eps_tag = str(args.epsilon).replace(".", "p")
        out = out / f"{args.attack}_{args.defense}_eps{eps_tag}"
    out.mkdir(parents=True, exist_ok=True)
    # 保存本次运行的全部参数，便于之后核对 ε、码本大小及模型设置。
    # vars(args) 把 argparse 的各项参数转成字典，json.dump 写成 JSON 文件。
    with open(out / "experiment_config.json", "w") as cf:
        json.dump(vars(args), cf, indent=2)
    FPG = args.num_frames_per_gop
    # 例如 FPG=33 时，每增加一个 GOP 只引入 32 个新帧。
    frames_per_gop_excl_first = FPG - 1 

    # ===== 数据入口第 1 步：找路径，还没有加载视频像素 =====
    # data_dir 可以是目录，也可以是单个 .yuv 文件。
    # uvg_data.py 中的 find_uvg_sequences 返回 list[tuple]，例如：
    # [("Jockey", "D:\...\Jockey_720p.yuv")]。
    all_seqs = find_uvg_sequences(args.data_dir)
    # 如果传了 --sequences，就用列表推导式只保留指定名字的 (n,p) 元组。
    # n 是序列名、p 是路径；此处仍只处理文件名，不读取帧。
    if args.sequences: all_seqs = [(n, p) for n, p in all_seqs if n in args.sequences]
    # 没有找到有效 .yuv 时停止运行。
    if not all_seqs: sys.exit(1)

    # ================================================================
    # 模型加载与双 GPU 内存分配
    # ================================================================
    # 注意执行顺序：本段在真正读取 YUV 之前运行，所以入口被这大段代码隔开。
    # 首次阅读数据流程时可以先跳到下方的“数据入口第 2 步”。
    model = WanFLF2VWrapper(args.wan_ckpt, config_name="flf2v-14B", flow_shift=args.flow_shift)
    print("\n[GVCC] Loading 14B model safely to CPU first...")
    # 先在 CPU 上以 BF16 加载；下方再把 DiT 的不同模块分配到两张 GPU。
    model.load("cpu", torch.bfloat16)
    
    print("[GVCC] Activating Hugging Face Accelerate for Dual RTX 5090s...")
    
    # 这是自动设备映射的显存上限，不是运行时的实际占用量。
    # GPU 0 主要放 DiT；GPU 1 留出更多空间给临时调用的 VAE / 文本 / CLIP。
    max_mem = {0: "22GiB", 1: "9GiB"}  
    # dict 的键 0/1 是 GPU 编号；Accelerate 根据上限生成模块→设备映射。
    device_map = infer_auto_device_map(model.model, max_memory=max_mem)
    
    # 把同一个 Transformer block 的子模块合并到同一设备，避免跨卡切分 block；
    # 映射里其余未匹配 block 的条目指定给 GPU 0。
    clean_map = {}
    # .items() 遍历字典的 (键, 值)；这里 key 是模块名，value 是目标设备。
    for key, value in device_map.items():
        match = re.search(r'(^.*?blocks\.\d+)', key)
        if match:
            block_key = match.group(1)
            if block_key not in clean_map:
                clean_map[block_key] = value
        else:
            clean_map[key] = 0
            
    model.model = dispatch_model(model.model, device_map=clean_map)
    # 后续采样张量的主设备是 GPU 0；这不表示整个模型只占 GPU 0。
    model.device = torch.device("cuda:0")

    # 将 CLIP / 文本编码器 / VAE 的一次调用临时路由到 GPU 1。
    # 这些是运行期的包装函数，不修改网络结构或已加载的权重。
    print("[GVCC] Injecting Smart Memory Interceptors (Targeting Vacuum Zone GPU 1)...")
    
    if hasattr(model, 'clip') and model.clip is not None:
        # hasattr(x,"clip") 检查对象有没有 clip 属性；is not None 再确认它非空。
        # 保存原方法后用包装函数替代，调用前后负责模块/输入/输出的搬运。
        original_clip_visual = model.clip.visual
        # FLF2V 的首尾帧条件会使用 CLIP 视觉特征。
        def patched_clip_visual(videos, *args, **kwargs):
            # *args 接收多余位置参数；**kwargs 接收多余“名字=值”参数，
            # 原样转发给 original_clip_visual，避免改变原接口。
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
    # 即使传空字符串，encode_prompt 仍要产生文本条件 embedding。
    def patched_encode_prompt(prompt, *args, **kwargs):
        # 暂时把文本模型移到 GPU 1，得到 embedding 后恢复原主设备设置。
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
        # VAE 编码/解码也采用同样的临时跨卡方式，以控制显存峰值。
        original_vae_encode = model.vae.encode
        # 视频 / 边界条件编码时，VAE 暂驻 GPU 1；结果再返回 GPU 0。
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
        # 重建潜变量转 RGB 时同样临时使用 GPU 1。
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

    # 设备已由 Accelerate 分配；阻止后续包装器调用 .to() / .cpu()
    # 意外地整体搬动被切分的 14B DiT。这也是本脚本只支持当前推理流程的原因之一。
    model.model.cpu = lambda *args, **kwargs: model.model
    model.model.to = lambda *args, **kwargs: model.model
    print("[GVCC] All systems green. Asymmetric balancing activated (22:9 ratio).")
    # ====================================================

    # all_seq_results 是列表，用于最后汇总多条 UVG 序列的指标。
    all_seq_results = []
    # ===== 数据入口第 2 步：逐条序列读取真正的像素 =====
    # all_seqs 中每个元素是 (序列名, YUV路径)；括号左侧把它拆成两个变量。
    # enumerate 额外给出从 0 开始的 seq_idx，仅用于显示第几条序列。
    for seq_idx, (seq_name, yuv_path) in enumerate(all_seqs):
        print(f"\n{'='*70}\n  [{seq_idx+1}/{len(all_seqs)}] Sequence: {seq_name}\n{'='*70}")
        seq_dir = out / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        # N 个 33 帧 GOP 共需 32*N+1 个独立输入帧（相邻 GOP 共用边界帧）。
        # 如果 num_gops=1、FPG=33，total_unique_frames=33。
        # 条件表达式 A if 条件 else B：条件成立取 A，否则取 B。
        total_unique_frames = frames_per_gop_excl_first * args.num_gops + 1 if args.num_gops > 0 else 999999
        # ★ 真正读取视频数据的是这一行：传入 yuv_path，内部 open(...,'rb')、f.read(...)。
        # raw_frames 是按时间排序的 PIL.Image 列表，尚不是 PyTorch Tensor；
        # 第 0 项 raw_frames[0] 是一张 RGB 帧，大小由 YUV 文件名/默认值决定。
        raw_frames = load_yuv420_frames(yuv_path, total_unique_frames, start_frame=0)
        # 文件短于请求长度时缩减 GOP 数；num_gops<=0 则按实际读到的帧数估算。
        # len(raw_frames) 是实际成功读取的帧数；// 是整数除法。
        actual_gops = max(1, (len(raw_frames) - 1) // frames_per_gop_excl_first) if (args.num_gops <= 0 or len(raw_frames) < total_unique_frames) else args.num_gops
        actual_total = frames_per_gop_excl_first * actual_gops + 1
        # [:actual_total] 取前 actual_total 帧，右端不包含 actual_total 本身。
        raw_frames = raw_frames[:actual_total]
        # 后续攻防和质量计算都使用统一分辨率的 RGB 帧；原始 YUV 不再保留。
        # frames_resized 仍是 list[PIL.Image]；720p 情况每帧 1280×720×3。
        frames_resized = resize_frames(raw_frames, WIDTH, HEIGHT)
        # del 只是删掉 raw_frames 变量引用，避免同时保留两份大视频列表。
        del raw_frames

        # 两种预处理按顺序作用：原帧 → 攻击 → 防御 → 实际送入编码器。
        # frames_resized 始终保留干净版本，稍后作为 PSNR / LPIPS 的比较目标。
        # 返回两个值：被扰动帧列表与记录参数的元数据字典。
        attacked_frames, attack_metadata = apply_attack(
            frames_resized,
            args.attack,
            epsilon=args.epsilon,
            attack_steps=args.attack_steps,
            attack_alpha=args.attack_alpha,
            seed=args.seed,
            frames_per_gop_excl_first=frames_per_gop_excl_first,
        )
        # 防御收到的是“攻击后的帧”；如果 defense=none，仍经过统一接口。
        # codec_frames 才是后面被 VAE 编码、用于首尾帧条件的实际输入。
        codec_frames, defense_metadata = apply_defense(
            attacked_frames,
            args.defense,
            jpeg_quality=args.jpeg_quality,
            median_size=args.median_size,
        )

        # 分别保留“干净真值 GOP”和“攻防后编码输入 GOP”。
        # 第 g 个 GOP 的区间是 [g*(FPG-1), g*(FPG-1)+FPG)，边界帧会重复。
        # 例如 FPG=33：GOP0 取帧 0..32，GOP1 取帧 32..64。
        gops_gt = []
        gops_codec = []
        for g in range(actual_gops):
            start = g * frames_per_gop_excl_first
            end = start + FPG
            # [start:end] 取右端之前的帧；append 把一个 GOP 列表放到外层列表。
            # gops_gt[g] / gops_codec[g] 都是长度约 33 的 PIL 帧列表。
            gops_gt.append(frames_resized[start:end])
            gops_codec.append(codec_frames[start:end])

        # 保存可视化视频：original、attacked_input、codec_input 分别对应上述三阶段。
        save_video_mp4(frames_resized, seq_dir / "original.mp4")
        if args.attack != "none":
            save_video_mp4(attacked_frames, seq_dir / "attacked_input.mp4")
        if args.attack != "none" or args.defense != "none":
            save_video_mp4(codec_frames, seq_dir / "codec_input.mp4")
        with open(seq_dir / "preprocess_config.json", "w") as pf:
            json.dump({"attack": attack_metadata, "defense": defense_metadata}, pf, indent=2)

        # FLF2V 使用每个 GOP 的首、尾两帧作为共享条件，而非把完整视频提供给 DiT。
        # 相邻 GOP 共用一张边界帧；在序列层面只压缩并计费一次。
        # 对 1 个 GOP、33 帧：boundary_indices=[0,32]。
        boundary_indices = [min(g * frames_per_gop_excl_first, len(codec_frames) - 1) for g in range(actual_gops + 1)]
        # 字典：帧下标 → (边界帧解压后的 PIL 图像, 压缩字节数)。
        boundary_compressed = {}
        total_boundary_bytes = 0

        for idx in boundary_indices:
            gt_frame = codec_frames[idx]
            # gt 是不压缩边界帧的消融模式，边界码率记为 0，不代表现实中免费传输。
            if args.ref_codec == "gt":
                boundary_compressed[idx] = (gt_frame, 0)
            else:
                # FLF2V 两端使用相同的“解压后”帧作条件，而不是让解码端看到原始帧。
                decoded, nbytes = compress_boundary_frame(gt_frame, ref_codec=args.ref_codec, ref_quality=args.ref_quality)
                boundary_compressed[idx] = (decoded, nbytes)
                total_boundary_bytes += nbytes

        gop_results = []
        all_recon_frames = []

        # 外层 for 处理序列；这里的内层 for 再逐个处理这个序列的 GOP。
        for g in range(actual_gops):
            print(f"\n  --- {seq_name} GOP {g}/{actual_gops-1}  ({datetime.now().strftime('%H:%M:%S')}) ---")
            gop_frames_gt = gops_gt[g]
            # gop_frames_gt 是干净评估目标；gop_frames 是攻防后的编码输入。
            gop_frames = gops_codec[g]
            first_idx, last_idx = g * frames_per_gop_excl_first, (g + 1) * frames_per_gop_excl_first
            first_decoded, first_bytes = boundary_compressed[first_idx]
            last_decoded, last_bytes = boundary_compressed[last_idx]

            # 前一 GOP 的尾帧也是后一 GOP 的首帧，所以只在 GOP 0 计首帧字节数。
            gop_first_bytes = 0 if g > 0 else first_bytes
            # pipeline 持有码本、采样时间表和噪声种子；不重新训练生成模型。
            # K=16384 是每次可选的噪声原子总数，M=64 是选中的原子数。
            # 此对象在每个 GOP 新建一次，使 GOP 之间使用一致的实验参数。
            pipe = TurboDDCMWanPipeline(
                model, K=args.K, M=args.M, num_steps=args.steps, num_ddim_tail=args.ddim_tail,
                guidance_scale=1.0, g_scale=args.g_scale, num_frames=FPG,
                height=HEIGHT, width=WIDTH, seed=args.seed,
            )
            
            # 将解压后的两张边界 RGB 帧变成 FLF2V 条件。
            # wrapper 中：CLIP 提取两帧视觉特征；VAE 编码“首帧+零帧+尾帧”，
            # 再加首尾位置掩码，返回包含 clip_fea 和 y 的字典。
            # 33 帧、720p 时 y 的示例形状是 [20,9,90,160]；
            # 其中 16 个 VAE 通道 + 4 个掩码通道，时间和空间与视频潜变量对齐。
            flf2v_cond = model.encode_first_last_frames(first_decoded, last_decoded, FPG, HEIGHT, WIDTH)
            
            # 编码端可见完整的攻防后 GOP：VAE 得到 x0_true，再逐步搜索码本索引/符号。
            print(f"  Encoding...")
            t0 = time.time()
            # 右侧函数返回两个对象，左侧两个变量一次接收（解包赋值）。
            # step_data 保存 [SDE步][潜在时间帧] 的码本索引/符号；
            # x0_true 是编码端的 [1,16,9,90,160] 真值潜变量，解码不用传它。
            step_data, x0_true = flf2v_encode(pipe, model, gop_frames, flf2v_cond, HEIGHT, WIDTH)
            t_enc = time.time() - t0

            # 编码阶段结束后释放条件和真值潜变量，给解码阶段留显存。
            del flf2v_cond, x0_true
            gc.collect()
            torch.cuda.empty_cache()

            print(f"  Decoding...")
            t0 = time.time()
            # 解码端只用相同边界条件、共享种子及 step_data 重放轨迹；不读原视频。
            # 这里是内存中的编解码往返：直接传 step_data，而非重新打开 .tdcm。
            # 单独再次构造条件，是为了模拟编解码器两侧各自计算条件特征。
            flf2v_cond_dec = model.encode_first_last_frames(first_decoded, last_decoded, FPG, HEIGHT, WIDTH)
            frames_recon = flf2v_decode(pipe, model, step_data, flf2v_cond_dec)
            t_dec = time.time() - t0

            gop_dir = seq_dir / f"gop{g}"
            gop_dir.mkdir(parents=True, exist_ok=True)
            # reconstructed.mp4 是可直接观看的重建视频；codebook.tdcm 是码本码流。
            # 指标使用内存里的 frames_recon，不使用 MP4 重新解码后的帧。
            save_video_mp4(frames_recon, gop_dir / "reconstructed.mp4")
            pipe.save_compressed(step_data, str(gop_dir / "codebook.tdcm"))

            # 拼接整条序列时跳过后一 GOP 重复的首帧；单个 GOP 文件仍保留 33 帧。
            # extend 会把列表中的每帧逐一追加；[1:] 跳过第 0 帧。
            if g == 0: all_recon_frames.extend(frames_recon)
            else: all_recon_frames.extend(frames_recon[1:])

            # 指标是当前 GOP 全部帧对干净真值的统计，而不是只比较某一张展示帧。
            # 这也是为什么单帧截图看起来差异小，整体指标却可能变化。
            n = min(len(gop_frames_gt), len(frames_recon))
            # 两个长度可能不完全相同；只取共同的前 n 帧再转 [n,3,H,W] Tensor。
            t_gt, t_rec = frames_to_tensor(gop_frames_gt[:n]), frames_to_tensor(frames_recon[:n])

            mean_psnr, per_frame_psnr = compute_psnr(t_gt, t_rec)
            mean_msssim = compute_msssim(t_gt, t_rec)
            mean_lpips = compute_lpips(t_gt, t_rec, device="cuda:1")

            # BPP 用理论码本比特数 + 两侧边界帧压缩字节数计算。
            # 例如 20 步、尾部 3 步、33 帧时：17 个 SDE 步 × 9 个潜在时间帧。
            # 这里不直接用 .tdcm 文件大小；文件头等封装开销可能使实际文件更大。
            # K=16384 时索引需 14 bit，另加 1 bit 符号；M=64 时每潜在帧/步=960 bit。
            T_sde, F_lat = pipe.num_sde_steps, pipe.num_latent_frames
            codebook_bytes = (T_sde * F_lat * pipe.codebook.bits_per_frame_step) // 8
            gop_boundary_bytes = gop_first_bytes + last_bytes
            gop_total_bytes = codebook_bytes + gop_boundary_bytes
            # bit / (原 RGB 视频的总像素数)，分母是 33×720×1280，不是潜空间大小。
            bpp = (gop_total_bytes * 8) / (FPG * HEIGHT * WIDTH)
            
            # 每 GOP 单独保存指标，尤其 per_frame_PSNR_dB 可定位关键帧或局部退化。
            # result 是字典；键如 "PSNR_dB" 供 JSON、汇总脚本读取。
            result = {
                "sequence": seq_name,
                "gop": g,
                "attack": args.attack,
                "defense": args.defense,
                "epsilon": args.epsilon,
                "PSNR_dB": round(mean_psnr, 2),
                "per_frame_PSNR_dB": [round(float(v), 4) for v in per_frame_psnr.cpu().tolist()],
                "LPIPS": round(mean_lpips, 4),
                "MS_SSIM": None if mean_msssim is None else round(mean_msssim, 6),
                "BPP": round(bpp, 6),
                "gop_total_bytes": gop_total_bytes,
                "codebook_bytes": codebook_bytes,
                "boundary_bytes": gop_boundary_bytes,
                "encode_seconds": round(t_enc, 3),
                "decode_seconds": round(t_dec, 3),
            }
            gop_results.append(result)
            # 每个 gopN/metrics.json 记录整 GOP 平均指标与逐帧 PSNR。
            with open(gop_dir / "metrics.json", "w") as mf: json.dump(result, mf, indent=2)

            print(f"    PSNR={mean_psnr:.2f} dB, LPIPS={mean_lpips:.4f}, BPP={bpp:.6f}")
            del pipe, frames_recon, t_gt, t_rec, step_data, flf2v_cond_dec
            gc.collect()
            torch.cuda.empty_cache()

        # 多 GOP 时输出无重复边界帧的整段重建视频。
        save_video_mp4(all_recon_frames, seq_dir / "reconstructed_full.mp4")
        all_seq_results.append({"sequence": seq_name, "gop_results": gop_results})
        del frames_resized, attacked_frames, codec_frames, all_recon_frames
        gc.collect()

    # 终端只汇总每 GOP 的 PSNR / LPIPS；更多细节见相应 metrics.json。
    print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
    for sr in all_seq_results:
        for r in sr["gop_results"]:
            print(f"Seq: {r['sequence']} | GOP: {r['gop']} | PSNR: {r['PSNR_dB']} | LPIPS: {r['LPIPS']}")

if __name__ == "__main__":
    main()
