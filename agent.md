# STAMP Agent 上下文

创建时间：2026-05-24

这个文件用于保存对话中 Image #1 提供的工作上下文，方便后续 agent 在修改代码前快速理解这个仓库。

## 核心思想

STAMP 对 Qwen2-VL 做了改造，把“对话生成”和“分割 mask 预测”放进同一个流程里。

模型仍然像普通多模态大语言模型一样自回归生成文本。但是一旦生成到特殊 token `<|seg|>`，STAMP 就会切换到分割模式：

1. 在 `<|seg|>` 位置停止，并保留对应位置的 KV cache。
2. 插入一串 `<|mask|>` 查询 token。
3. 使用 `do_classification=True` 再跑一次前向。
4. 用每个 `<|mask|>` 的 hidden state 预测一个前景/背景 logit。
5. 把这些 logits reshape 成低分辨率 mask。
6. 可选地使用 SAM 对粗 mask 做边界细化。

关键区别在于：STAMP 不是逐像素、也不是逐 token 地慢慢生成 mask。它会插入大量 `<|mask|>` 查询 token，然后通过分类头并行预测整张 mask。

## 核心机制

核心实现主要在：

- `model/qwen_changes.py`
- `segment_predictor_cache.py`

预期机制如下：

- `SegQwenVL` 继承或包装 Qwen2-VL。
- 新增 `classifier = Linear(hidden_size, 1)`。
- 这个分类器把每个 mask query 的 hidden state 映射成一个二分类 logit。
- 推理时先使用普通 `generate` 生成文本。
- 在生成序列中找到 `<|seg|>` 后，在对应位置截断或复用 KV cache。
- 系统根据视觉 patch 网格创建 `<|mask|>` token。
- 第二次前向会为所有 mask query 位置预测前景/背景 logits。
- 这些 logits 被 reshape 成低分辨率 mask，并且可以交给 SAM 进一步细化。

## 训练逻辑

需要重点关注的训练入口是：

- `train/seg_trainer.py`

这个 trainer 不只是优化语言建模 loss，而是组合了：

- 语言建模 loss
- 分割 loss

截图中提到的分割 loss 是 `WeightedDiceBCELoss`，也就是 BCE loss 和 Dice loss 的组合，用来监督 `<|mask|>` 位置预测出的二值 mask。

简化理解：

```text
总 loss = 语言建模 loss + 分割 loss
分割 loss = BCE + Dice
```

## 数据流

整体 referring segmentation 流程可以概括为：

1. 输入图片和文本指令，例如：`segment the white horse`。
2. Qwen2-VL 生成回答，并在需要分割的位置输出 `<|seg|>`。
3. STAMP 插入大量 `<|mask|>` 查询 token。
4. 模型在一次前向中预测所有图像 patch 位置的前景概率。
5. 原始 logits 被 reshape 成粗 mask。
6. `model/segment_anything/` 可以选择性地使用 SAM 细化 mask 边界。
7. 使用 RefCOCO、RefCOCO+、RefCOCOg、gRefCOCO 等数据集评估，主要指标是 IoU。

## 主要仓库模块

- `readme.md`：项目说明、安装、推理、训练和数据准备。
- `run_seg_ref.py`：单图或 referring segmentation 推理入口。
- `segment_predictor_cache.py`：核心推理封装，包括 `GenerativeSegmenter`。
- `model/qwen_changes.py`：STAMP 对 Qwen2-VL 的关键改造。
- `model/modeling_qwen2_vl.py`：本地 Qwen2-VL 实现。
- `model/segment_anything/`：SAM 相关代码，用于 mask 后处理。
- `train/seg_trainer.py`：自定义 SFT trainer，加入分割监督。
- `dataset/` 和 `data/`：RefCOCO/gRefCOCO 数据读取与预处理。
- `eval/eval_refer_seg.py`：分割评估入口。
- `scripts/`：训练和评估脚本。
- `paper/2512.00395v1.pdf`：对话中提到的本地论文文件。

## 长期观测说明

这个文件是后续在本仓库中工作的持久观测日志。对话结束后不会有后台进程持续监控仓库；后续 agent 应该把这个文件当作第一个上下文文件来阅读，并在验证或修改相关行为后更新它。

观测或修改 STAMP 时，需要持续关注：

- `<|seg|>` 检测是否正确定位到生成序列中的分割触发位置。
- 第二次前向使用的 KV cache 是否和 `<|seg|>` 位置对齐。
- 插入的 `<|mask|>` token 数量是否匹配视觉 patch 网格或预期 mask 分辨率。
- `do_classification=True` 是否只返回 mask query 位置对应的 logits。
- 分类器输出形状是否和 mask reshape 逻辑一致。
- 训练标签和预测 mask logits 是否在空间位置上对齐。
- `WeightedDiceBCELoss` 是否按预期处理前景/背景不均衡。
- SAM refinement 是否是可选的，并且不会掩盖粗 mask 预测阶段的问题。
- RefCOCO、RefCOCO+、RefCOCOg、gRefCOCO 上的 IoU 评估是否和仓库预期协议一致。

## 下一步验证目标

下一次有效的代码阅读应该检查这些具体问题：

1. 在 `segment_predictor_cache.py` 中，`<|seg|>` 是在哪里从生成结果中定位的？第二次前向使用的 cache 或 index 是如何选择的？
2. 在 `model/qwen_changes.py` 中，分类器之前是如何选取 `<|mask|>` hidden states 的？
3. 在 `model/qwen_changes.py` 中，分类器返回的具体 tensor 形状是什么？在哪里被 reshape 成 mask？
4. 在 `train/seg_trainer.py` 中，语言建模 loss 和分割 loss 是如何组合、如何加权的？
5. 在 dataset 相关代码中，ground-truth mask 是如何 resize 或对齐到预测的低分辨率 mask 网格的？

## 状态日志

- 2026-05-24：已根据 Image #1 初始化上下文。尚未验证具体代码行为。
