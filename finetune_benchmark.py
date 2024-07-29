import os
import sys
import time
from typing import List
import numpy as np

import fire
import torch
import transformers
from datasets import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import LlamaForCausalLM

def benchmark(
    # model/data params
    base_model: str = "",  # the only required argument
    sample_total: int = 5120,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"sample_total: {sample_total}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"group_by_length: {group_by_length}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    mock_data = {
                "instruction": [f"instruction{i}" for i in range(sample_total)],
                "input": [f"input{i}" for i in range(sample_total)],
                "output": [f"output{i}" for i in range(sample_total)],
                "input_ids": np.random.randint(100, size=(sample_total, cutoff_len)),
                "attention_mask": np.ones((sample_total, cutoff_len)),
                "labels": np.random.randint(100, size=(sample_total, cutoff_len))
            }
    mock_train_data = Dataset.from_dict(mock_data)
    sample_count = mock_train_data.num_rows

    print(f"Training on {sample_count} samples, on {sample_count * cutoff_len} tokens.")

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=mock_train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            output_dir="./mock_model",
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    start = time.time()
    trainer.train()
    during = time.time() - start
    throughput_sample = sample_count / during
    throughput_token = sample_count * cutoff_len / during
    print(f"Training throughput_samples: {throughput_sample:.2f} samples/s, throughput_token: {throughput_token:.2f} tokens/s")

if __name__ == "__main__":
    fire.Fire(benchmark)
