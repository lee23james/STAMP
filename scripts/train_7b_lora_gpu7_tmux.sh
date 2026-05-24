#!/usr/bin/env bash
set -euo pipefail

cd /home/honghudata/deepseek_VG/routing/STAMP

export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/home/honghudata/deepseek_VG/routing/STAMP
export HF_HOME=/home/honghudata/deepseek_VG/routing/STAMP/.hf_cache
export HF_HUB_CACHE=/home/honghudata/deepseek_VG/routing/STAMP/.hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

export STAMP_7B_BASE_MODEL=/home/honghudata/deepseek_VG/maker/dataset/Qwen2-VL-7B-Instruct
if [[ -z "${STAMP_INIT_ADAPTER+x}" ]]; then
  export STAMP_INIT_ADAPTER=/home/honghudata/deepseek_VG/maker/STAMP/checkpoints/STAMP-7B-lora
elif [[ "${STAMP_INIT_ADAPTER}" == "none" ]]; then
  export STAMP_INIT_ADAPTER=
fi
export STAMP_TRAIN_JSON_ROOT=/home/honghudata/deepseek_VG/maker/STAMP/train/json_files
export STAMP_TRAIN_IMAGE_ROOT=/home/honghudata/deepseek_VG/maker/dataset/refer_seg_sesame
export STAMP_TRAIN_MASK_ROOT=/home/honghudata/deepseek_VG/maker/STAMP
export STAMP_TRAIN_OUTPUT_DIR="${STAMP_TRAIN_OUTPUT_DIR:-/home/honghudata/deepseek_VG/routing/STAMP/output/train_7b_lora_local}"

export STAMP_BATCH_SIZE="${STAMP_BATCH_SIZE:-1}"
export STAMP_GRAD_ACCUM="${STAMP_GRAD_ACCUM:-8}"
export STAMP_LEARNING_RATE="${STAMP_LEARNING_RATE:-1e-5}"
export STAMP_TORCH_DTYPE="${STAMP_TORCH_DTYPE:-fp32}"
export STAMP_BF16="${STAMP_BF16:-0}"
export STAMP_FP16="${STAMP_FP16:-0}"
export STAMP_LOGGING_STEPS="${STAMP_LOGGING_STEPS:-1}"
export STAMP_SAVE_STEPS="${STAMP_SAVE_STEPS:-1000}"
export STAMP_MAX_TRAIN_SAMPLES="${STAMP_MAX_TRAIN_SAMPLES:-0}"
export STAMP_MAX_STEPS="${STAMP_MAX_STEPS:--1}"

mkdir -p output_eval/logs output/train_7b_lora_local

echo "[$(date '+%F %T %Z')] Starting local 7B LoRA training on GPU 7"
echo "STAMP_BATCH_SIZE=${STAMP_BATCH_SIZE}"
echo "STAMP_GRAD_ACCUM=${STAMP_GRAD_ACCUM}"
echo "STAMP_LEARNING_RATE=${STAMP_LEARNING_RATE}"
echo "STAMP_INIT_ADAPTER=${STAMP_INIT_ADAPTER}"
echo "STAMP_TORCH_DTYPE=${STAMP_TORCH_DTYPE}"
echo "STAMP_BF16=${STAMP_BF16}"
echo "STAMP_FP16=${STAMP_FP16}"
echo "STAMP_TRAIN_OUTPUT_DIR=${STAMP_TRAIN_OUTPUT_DIR}"
echo "STAMP_MAX_TRAIN_SAMPLES=${STAMP_MAX_TRAIN_SAMPLES}"
echo "STAMP_MAX_STEPS=${STAMP_MAX_STEPS}"

conda run -n STAMP python -m train.main_seg_train_7B_lora_local
