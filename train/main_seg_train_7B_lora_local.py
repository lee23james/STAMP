import json
import os
from typing import Any, Dict, List

import deepspeed
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from trl import SFTConfig
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

from model.qwen_changes import SegQwenVL, get_rope_index
from train.seg_trainer import SegmentationSFTTrainer

deepspeed.ops.op_builder.CPUAdamBuilder().load()

IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "-1") in ["-1", "0"]


def _resolve_torch_dtype(value: str):
    value = value.lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported STAMP_TORCH_DTYPE={value!r}")


class CustomLogCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if hasattr(self.trainer, "_custom_log_metrics") and self.trainer._custom_log_metrics:
            logs = kwargs.get("logs", {})
            logs.update(self.trainer._custom_log_metrics)
            self.trainer._custom_log_metrics = {}


class CustomDataCollator(DataCollatorForVisionLanguageModeling):
    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        all_masks = []
        all_images = []
        for example in examples:
            all_images.append([Image.open(path).convert("RGB") for path in example["images"]])
            if "masks" in example and example["masks"] is not None:
                all_masks.append([Image.open(path).convert("L") for path in example["masks"]])
            else:
                all_masks.append([])

        batch["masks"] = all_masks
        batch["all_images"] = all_images
        return batch


class QwenVL7BLoRALocalTrainer:
    def __init__(self):
        self.model_name = os.environ.get(
            "STAMP_7B_BASE_MODEL",
            "/home/honghudata/deepseek_VG/maker/dataset/Qwen2-VL-7B-Instruct",
        )
        self.output_dir = os.environ.get(
            "STAMP_TRAIN_OUTPUT_DIR",
            "/home/honghudata/deepseek_VG/routing/STAMP/output/train_7b_lora_local",
        )
        self.json_root = os.environ.get(
            "STAMP_TRAIN_JSON_ROOT",
            "/home/honghudata/deepseek_VG/maker/STAMP/train/json_files",
        )
        self.image_root = os.environ.get(
            "STAMP_TRAIN_IMAGE_ROOT",
            "/home/honghudata/deepseek_VG/maker/dataset/refer_seg_sesame",
        )
        self.mask_root = os.environ.get(
            "STAMP_TRAIN_MASK_ROOT",
            "/home/honghudata/deepseek_VG/maker/STAMP",
        )
        self.init_adapter = os.environ.get("STAMP_INIT_ADAPTER", "").strip()
        self.max_train_samples = int(os.environ.get("STAMP_MAX_TRAIN_SAMPLES", "0"))

        if IS_MAIN_PROCESS:
            print("--- Loading 7B base model and processor for local LoRA training ---", flush=True)
            print(f"base_model={self.model_name}", flush=True)
            print(f"init_adapter={self.init_adapter or '<from scratch>'}", flush=True)
            print(f"json_root={self.json_root}", flush=True)
            print(f"image_root={self.image_root}", flush=True)
            print(f"mask_root={self.mask_root}", flush=True)
            print(f"output_dir={self.output_dir}", flush=True)

        min_pixels = int(os.environ.get("STAMP_MIN_PIXELS", str(1024 * 28 * 28)))
        max_pixels = int(os.environ.get("STAMP_MAX_PIXELS", str(1280 * 28 * 28)))
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            local_files_only=True,
        )

        model_dtype = _resolve_torch_dtype(os.environ.get("STAMP_TORCH_DTYPE", "bf16"))
        self.model = SegQwenVL.from_pretrained(
            self.model_name,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True,
            quantization_config=None,
        )
        type(self.model.model).get_rope_index = get_rope_index

        special_tokens = {"additional_special_tokens": ["<|seg|>", "<|mask|>", "<|yes|>", "<|no|>"]}
        num_added_tokens = self.processor.tokenizer.add_special_tokens(special_tokens)
        if num_added_tokens > 0:
            if IS_MAIN_PROCESS:
                print(f"--- Added {num_added_tokens} segmentation tokens ---", flush=True)
            self.model.resize_token_embeddings(len(self.processor.tokenizer))

        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id

        self.collator = CustomDataCollator(processor=self.processor)
        self.model.mask_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|mask|>")

    def _resolve_image_path(self, path: str) -> str:
        path = path.replace("refer_seg/", "")
        path = path.replace("images/coco_2014/", "images/mscoco/images/")
        return os.path.join(self.image_root, path)

    def _resolve_mask_path(self, path: str) -> str:
        return os.path.join(self.mask_root, path)

    def _create_dataset(self):
        json_files = [
            "refclef_formatted_all_sentences_doubled_mp.json",
            "refcocog_formatted_all_sentences_doubled_mp.json",
            "refcoco_formatted_all_sentences_doubled_mp.json",
            "refcoco+_formatted_all_sentences_doubled_mp.json",
        ]

        all_data = []
        for name in json_files:
            path = os.path.join(self.json_root, name)
            if IS_MAIN_PROCESS:
                print(f"--- Loading {path} ---", flush=True)
            with open(path, "r", encoding="utf-8") as file:
                all_data.extend(json.load(file))

        if self.max_train_samples > 0:
            all_data = all_data[: self.max_train_samples]

        processed_data = []
        missing = 0
        for example in tqdm(all_data, disable=not IS_MAIN_PROCESS):
            example["images"] = [self._resolve_image_path(path) for path in example["images"]]
            if "masks" not in example or example["masks"] is None:
                example["masks"] = None
            else:
                example["masks"] = [self._resolve_mask_path(path) for path in example["masks"]]

            paths_to_check = list(example["images"]) + ([] if example["masks"] is None else list(example["masks"]))
            if all(os.path.exists(path) for path in paths_to_check):
                processed_data.append(example)
            else:
                missing += 1

        if IS_MAIN_PROCESS:
            print(f"--- Dataset records loaded: {len(all_data)}; usable: {len(processed_data)}; skipped_missing: {missing} ---", flush=True)
            print(f"--- Sample messages: {processed_data[0]['messages']} ---", flush=True)

        return Dataset.from_list(processed_data)

    def train(self):
        train_dataset = self._create_dataset()

        train_lm_head = os.environ.get("STAMP_TRAIN_LM_HEAD", "1") != "0"
        modules_to_save = ["classifier"]
        if train_lm_head:
            modules_to_save = ["embed_tokens", "lm_head", "classifier"]

        peft_config = LoraConfig(
            r=int(os.environ.get("STAMP_LORA_R", "64")),
            lora_alpha=int(os.environ.get("STAMP_LORA_ALPHA", "128")),
            lora_dropout=float(os.environ.get("STAMP_LORA_DROPOUT", "0.05")),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_rslora=True,
            modules_to_save=modules_to_save,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if self.init_adapter:
            if IS_MAIN_PROCESS:
                print(f"--- Loading trainable LoRA adapter from {self.init_adapter} ---", flush=True)
            self.model = PeftModel.from_pretrained(
                self.model,
                self.init_adapter,
                is_trainable=True,
                local_files_only=True,
            )
        else:
            self.model = get_peft_model(self.model, peft_config)
        if IS_MAIN_PROCESS:
            self.model.print_trainable_parameters()

        max_steps = int(os.environ.get("STAMP_MAX_STEPS", "-1"))
        training_args = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=float(os.environ.get("STAMP_NUM_TRAIN_EPOCHS", "5")),
            max_steps=max_steps,
            per_device_train_batch_size=int(os.environ.get("STAMP_BATCH_SIZE", "1")),
            gradient_accumulation_steps=int(os.environ.get("STAMP_GRAD_ACCUM", "8")),
            learning_rate=float(os.environ.get("STAMP_LEARNING_RATE", "3e-5")),
            lr_scheduler_type="linear",
            warmup_ratio=float(os.environ.get("STAMP_WARMUP_RATIO", "0.03")),
            weight_decay=0.0,
            optim=os.environ.get("STAMP_OPTIM", "adamw_torch_fused"),
            max_grad_norm=1.0,
            logging_steps=int(os.environ.get("STAMP_LOGGING_STEPS", "1")),
            save_steps=int(os.environ.get("STAMP_SAVE_STEPS", "1000")),
            bf16=os.environ.get("STAMP_BF16", "1") != "0",
            fp16=os.environ.get("STAMP_FP16", "0") == "1",
            tf32=True,
            remove_unused_columns=False,
            report_to="none",
            max_length=None,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        trainer = SegmentationSFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=self.processor,
            data_collator=self.collator,
        )

        if trainer.is_world_process_zero():
            print("--- Starting 7B LoRA training ---", flush=True)
        trainer.train()
        if trainer.is_world_process_zero():
            print("--- Training finished ---", flush=True)

        final_model_path = os.path.join(self.output_dir, "final_model")
        trainer.save_model(final_model_path)
        if trainer.is_world_process_zero():
            print(f"--- Model saved to {final_model_path} ---", flush=True)


if __name__ == "__main__":
    QwenVL7BLoRALocalTrainer().train()
