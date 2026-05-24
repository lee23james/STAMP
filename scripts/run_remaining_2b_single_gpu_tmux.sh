#!/usr/bin/env bash
set -euo pipefail

cd /home/honghudata/deepseek_VG/routing/STAMP

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/honghudata/deepseek_VG/routing/STAMP/.hf_cache
export HF_HUB_CACHE=/home/honghudata/deepseek_VG/routing/STAMP/.hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/home/honghudata/deepseek_VG/routing/STAMP

MODEL_PATH="/home/honghudata/deepseek_VG/routing/STAMP/checkpoints/STAMP-2B-uni"
SAM_PATH="sam_vit_h_4b8939.pth"
IMAGE_FOLDER="/home/honghudata/deepseek_VG/maker/dataset/refer_seg_sesame"

mkdir -p output_eval/logs

for split in \
  "refcoco+|unc|testB" \
  "refcocog|umd|val" \
  "refcocog|umd|test"
do
  echo "=================================================="
  echo "[$(date '+%F %T %Z')] Evaluating ${split} on single GPU"
  echo "=================================================="

  out="output_eval/2b_full_single_gpu/${split//|/_}/"
  CUDA_VISIBLE_DEVICES=6 conda run -n STAMP python eval/eval_refer_seg.py \
    --model_path "${MODEL_PATH}" \
    --sam_path "${SAM_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --dataset_split "${split}" \
    --save_file "${out}"

  echo "[$(date '+%F %T %Z')] Finished ${split}"
done
