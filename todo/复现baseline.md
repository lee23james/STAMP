# STAMP baseline 复现记录

## 任务理解

学长的要求是：先使用 STAMP 官方仓库中的预测代码和作者公开模型参数跑一次 baseline，基本确认论文方法能复现，而不是马上改模型或重新训练。

本阶段目标：

1. 使用官方推理入口 `run_seg_ref.py`。
2. 使用作者公开权重 `JiaZL/STAMP-2B-uni`。
3. 使用 SAM 权重做 mask refine。
4. 先跑通单图预测 demo。
5. 记录命令、环境、输出和 mask 基本统计。
6. 后续再进入 RefCOCO / RefCOCO+ / RefCOCOg 的 gIoU / cIoU 评估。

## 当前环境

- 仓库路径：`/home/honghudata/deepseek_VG/routing/STAMP`
- Conda 环境：`STAMP`
- Python：`3.10.20`
- PyTorch：`2.9.0+cu126`
- CUDA used by torch：`12.6`
- Transformers：`4.57.1`
- Accelerate：`1.11.0`
- PEFT：`0.17.1`
- `flash-attn`：未安装

说明：

- 原计划中 `flash-attn` 安装失败，原因是 GitHub wheel 下载连接被远端关闭。
- 用户随后确认：先不安装 `flash-attn`，直接运行单图推理。
- 当前 `segment_predictor_cache.py` 中 `attn_implementation="flash_attention_2"` 是注释状态，因此本次单图推理未触发 `flash-attn` 报错。

## 已复用资源

SAM 权重没有重新下载，而是使用 maker 目录中的已有文件，并在当前仓库根目录创建软链接：

```text
/home/honghudata/deepseek_VG/routing/STAMP/sam_vit_h_4b8939.pth
-> /home/honghudata/deepseek_VG/maker/STAMP/sam_vit_h_4b8939.pth
```

模型权重通过 hf-mirror 下载：

```text
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=/home/honghudata/deepseek_VG/routing/STAMP/.hf_cache
HF_HUB_CACHE=/home/honghudata/deepseek_VG/routing/STAMP/.hf_cache/hub
```

下载模型：

```text
JiaZL/STAMP-2B-uni
```

下载完成后的 snapshot 路径：

```text
/home/honghudata/deepseek_VG/routing/STAMP/.hf_cache/hub/models--JiaZL--STAMP-2B-uni/snapshots/c825248230b11e63a58286e4a4c5d4a7dbbbe0f3
```

## 2026-05-24 单图预测记录

### 运行命令

```bash
CUDA_VISIBLE_DEVICES=2,3 \
HF_ENDPOINT=https://hf-mirror.com \
HF_HOME=/home/honghudata/deepseek_VG/routing/STAMP/.hf_cache \
HF_HUB_CACHE=/home/honghudata/deepseek_VG/routing/STAMP/.hf_cache/hub \
conda run -n STAMP python run_seg_ref.py \
    --model-path "JiaZL/STAMP-2B-uni" \
    --image-file "images/horses.png" \
    --sam_path "sam_vit_h_4b8939.pth" \
    --query "Please segment the white horse in the image."
```

### 实际运行情况

- 运行状态：成功
- 使用模型：`JiaZL/STAMP-2B-uni`
- 输入图片：`images/horses.png`
- Query：`Please segment the white horse in the image.`
- SAM：`sam_vit_h_4b8939.pth`
- 指定 GPU：`CUDA_VISIBLE_DEVICES=2,3`
- 实际主要占用：物理 GPU 2
- 输出文件：`STAMP/images/horses_mask.jpg`

说明：

- `run_seg_ref.py` 内部 `GenerativeSegmenter` 使用 `device_map="cuda"`，所以虽然暴露了 GPU 2、3，但单图推理主要使用可见设备中的第 0 张卡，也就是物理 GPU 2。
- 本次没有修改推理代码。

### 模型输出文本

```text
The segmentation mask for 'white horse' is shown below:
```

### mask 文件检查

输出文件：

```text
/home/honghudata/deepseek_VG/routing/STAMP/STAMP/images/horses_mask.jpg
```

文件大小：

```text
65K
```

像素统计：

```text
shape: (1367, 2048, 3)
dtype: uint8
min: 0
max: 255
mean: 12.140507126691661
nonzero: 412524
```

结论：

- mask 文件已生成。
- `max=255`，`nonzero=412524`，说明输出不是全黑。
- 单图 baseline 推理流程已跑通。

## 后续复现计划

下一步可以进入 RefCOCO 系列评估，目标是得到 gIoU / cIoU 并和论文表格对比。

已发现可复用数据路径：

```text
/home/honghudata/deepseek_VG/maker/dataset/refer_seg_sesame
```

已发现 maker 目录中有 7B LoRA baseline 资源：

```text
/home/honghudata/deepseek_VG/maker/STAMP/checkpoints/STAMP-7B-lora
/home/honghudata/deepseek_VG/maker/dataset/Qwen2-VL-7B-Instruct
```

maker 目录中已有一次 7B 在 `refcoco|unc|val` 上的评估结果：

```text
Raw Model Mask   -> gIoU: 0.7628, cIoU: 0.7796
SAM-Refined Mask -> gIoU: 0.8337, cIoU: 0.8305
```

建议下一步：

1. 先用 `JiaZL/STAMP-2B-uni` 跑 `refcoco|unc|val`。
2. 如果 2B 评估流程正常，再跑完整 RefCOCO / RefCOCO+ / RefCOCOg split。
3. 如需对齐论文 Table 2，再使用 maker 中已有的 `STAMP-7B-lora` 和 Qwen2-VL-7B base 跑 7B 评估。

## 2026-05-24 RefCOCO val 评估记录

### 运行前修复

运行评估前先按 maker 目录中的已有改动同步了两个必要修复：

1. `eval/eval_refer_seg.py`
   - 原始代码导入 `segment_predictor`，但当前仓库没有该文件。
   - maker 目录中的可运行版本使用 `segment_predictor_cache`。
   - 当前已同步为：

```python
from segment_predictor_cache import GenerativeSegmenter
```

2. `dataset/refer_seg_dataset.py`
   - 当前数据集图片实际位于：

```text
/home/honghudata/deepseek_VG/maker/dataset/refer_seg_sesame/images/mscoco/images/train2014
```

   - 原始代码查找的是：

```text
images/coco_2014/train2014
```

   - maker 目录中的可运行版本已改为：

```text
images/mscoco/images/train2014
```

   - 当前已同步该路径修复。

### 运行命令

第一次尝试误用了 `--gpu_ids "0,1"`，导致进程跑到物理 GPU 0、1，并因为已有 vLLM 服务占显存而 OOM。随后改为明确使用空闲物理 GPU 6、7：

```bash
HF_ENDPOINT=https://hf-mirror.com \
HF_HOME=/home/honghudata/deepseek_VG/routing/STAMP/.hf_cache \
HF_HUB_CACHE=/home/honghudata/deepseek_VG/routing/STAMP/.hf_cache/hub \
PYTHONPATH=/home/honghudata/deepseek_VG/routing/STAMP \
conda run -n STAMP accelerate launch \
    --num_processes=2 \
    --gpu_ids "6,7" \
    eval/eval_refer_seg.py \
    --model_path "JiaZL/STAMP-2B-uni" \
    --sam_path "sam_vit_h_4b8939.pth" \
    --image_folder "/home/honghudata/deepseek_VG/maker/dataset/refer_seg_sesame" \
    --dataset_split "refcoco|unc|val" \
    --save_file "output_eval/refcoco_unc_val_2b/"
```

### 结果

输出文件：

```text
output_eval/refcoco_unc_val_2b/STAMP-2B-uni/refcoco_unc_val/evaluation_results.txt
```

评估结果：

```text
[2026-05-24 12:54:29] Model: JiaZL/STAMP-2B-uni, Dataset: refcoco|unc|val
  - Evaluated on 1500 images and 10834 masks.
  - Raw Model Mask   -> gIoU: 0.7660, cIoU: 0.7783
  - SAM-Refined Mask -> gIoU: 0.8266, cIoU: 0.8212
```

结论：

- `JiaZL/STAMP-2B-uni` 在 `refcoco|unc|val` 上的评估流程已跑通。
- 数据读取、模型推理、SAM refine、双卡 accelerate、结果落盘均正常。
- 可以继续跑完整 RefCOCO / RefCOCO+ / RefCOCOg split。
