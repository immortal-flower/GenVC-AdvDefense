#!/bin/bash
# FLF2V Compression — 720p full UVG
# Uses Wan2.1 FLF2V-14B with first+last frame conditioning

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

python ${SCRIPT_DIR}/run_flf2v_experiment.py \
  --wan_ckpt ${SCRIPT_DIR}/Wan2.1-FLF2V-14B-720P \
  --data_dir ${PROJECT_DIR}/data/uvg \
  --output_dir ${SCRIPT_DIR}/results_720p \
  --num_frames_per_gop 33 \
  --num_gops 3 \
  --height 720 --width 1280 \
  --M 64 \
  --K 16384 \
  --steps 20 --ddim_tail 3 \
  --g_scale 3.0 \
  --ref_codec compressai --ref_quality 4 \
  --seed 42 \
  "$@"

# windows 命令
'''
python exp_flf2v/run_flf2v_experiment.py --wan_ckpt exp_flf2v/Wan2.1-FLF2V-14B-720P 
--data_dir "D:\yzb and lmk\dataset-720P\UVG" --output_dir exp_flf2v/results_720p
--num_frames_per_gop 33 --num_gops 3 --height 720 --width 1280 --M 64 --K 16384 
--steps 20 --ddim_tail 3 --g_scale 3.0 --ref_codec compressai --ref_quality 4 --seed 42
'''
'''
set CUDA_VISIBLE_DEVICES=0,1 && python exp_flf2v/run_flf2v_experiment.py
 --wan_ckpt exp_flf2v/Wan2.1-FLF2V-14B-720P --data_dir "D:\yzb and lmk\dataset-720P\UVG" --output_dir exp_flf2v/results_720p
  --num_frames_per_gop 33 --num_gops 3 --height 720 --width 1280 --M 64 --K 16384 --steps 20 --ddim_tail 3 --g_scale 3.0
   --ref_codec compressai --ref_quality 4 --seed 42
'''