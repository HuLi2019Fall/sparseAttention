import torch
import os
import time
import subprocess
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from itertools import chain
from modeling_qwen3 import Qwen3ForCausalLM
from configuration_qwen3 import Qwen3Config

# ------------------------------------------------------------
# 1. Setup
# ------------------------------------------------------------
model_id = "Qwen/Qwen3-0.6B"
dataset_path = "dataset/014_00000.parquet"
log_directory = "./logs"

os.makedirs(log_directory, exist_ok=True)

# Fix CUDA memory fragmentation (Helps prevent OOM on A4500)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set optimal defaults for A4500 (Ampere Architecture)
# TF32 = ~8x speedup on matrix math
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ------------------------------------------------------------
# 2. Tokenizer
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------------------
# 3. Dataset
# ------------------------------------------------------------
print(f"Loading dataset from {dataset_path}...")
full_dataset = load_dataset(
    "parquet",
    data_files={"train": dataset_path},
)["train"]

# Slice for testing (remove this line for full training)
raw_dataset = full_dataset.select(range(len(full_dataset) // 20))
print(f"Using {len(raw_dataset)} samples for processing.")

# ------------------------------------------------------------
# 4. Tokenization & Sequence Packing
# ------------------------------------------------------------
MAX_SEQ_LEN = 1024


def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True)


print("Tokenizing raw text...")
tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=raw_dataset.column_names,
    num_proc=8,
    load_from_cache_file=True,
)


def group_texts(examples):
    """Concatenates texts to eliminate padding efficiency loss."""
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= MAX_SEQ_LEN:
        total_length = (total_length // MAX_SEQ_LEN) * MAX_SEQ_LEN

    result = {
        k: [t[i : i + MAX_SEQ_LEN] for i in range(0, total_length, MAX_SEQ_LEN)]
        for k, t in concatenated_examples.items()
    }
    return result


print(f"Packing sequences into blocks of {MAX_SEQ_LEN} tokens...")
packed_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    num_proc=8,
    load_from_cache_file=True,
)

print(f"Packed samples: {len(packed_dataset)}")

# ------------------------------------------------------------
# 5. Model (Optimized for 20GB VRAM)
# ------------------------------------------------------------
print("Initializing model...")

# config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
config = Qwen3Config()

# Attempt to use Flash Attention 2
try:
    config._attn_implementation = "flash_attention_2"
    model = Qwen3ForCausalLM(
        config,
    )
    print("✓ Using Flash Attention 2")
except Exception as e:
    print(f"Flash Attention 2 not available: {e}")
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(
        config,
    )

model.to(torch.bfloat16)
model.to(device="cuda", dtype=torch.bfloat16)

# CRITICAL OPTIMIZATION: Disable Gradient Checkpointing
# model.gradient_checkpointing_disable()
model.config.use_cache = False

# PyTorch 2.0 Compilation
print("Compiling model with torch.compile (this takes ~60s at start)...")
try:
    model = torch.compile(model)
    print("✓ Model compiled")
except Exception as e:
    print(f"Compilation skipped: {e}")

# ------------------------------------------------------------
# 6. Data collator
# ------------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# ------------------------------------------------------------
# 6.5. Performance Profiling Callback (Verified)
# ------------------------------------------------------------
def get_nvidia_smi_memory():
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8",
        )
        memory_used, memory_total, gpu_util = result.strip().split(",")
        return (
            float(memory_used) / 1024,
            float(memory_total) / 1024,
            int(gpu_util),
        )
    except Exception:
        return None, None, None


class PerformanceCallback(TrainerCallback):
    def __init__(self, max_seq_len):
        self.step_times = []
        self.step_start_time = None
        self.total_tokens = 0
        self.max_seq_len = max_seq_len

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            self.step_times.append(step_time)

            batch_size = (
                args.per_device_train_batch_size * args.gradient_accumulation_steps
            )
            tokens_this_step = batch_size * self.max_seq_len
            self.total_tokens += tokens_this_step

            # Print stats every 10 steps
            if state.global_step % 10 == 0 and len(self.step_times) > 0:
                recent_times = self.step_times[-10:]
                avg_step_time = sum(recent_times) / len(recent_times)
                tokens_per_sec = tokens_this_step / avg_step_time

                print(f"\n[{state.global_step}/{args.max_steps}] Metrics:")
                print(f"  Throughput:    {tokens_per_sec:,.0f} tokens/sec")
                print(f"  Avg Step Time: {avg_step_time:.3f}s")

                memory_used, memory_total, util = get_nvidia_smi_memory()
                if memory_used:
                    print(
                        f"  GPU VRAM:      {memory_used:.1f}/{memory_total:.1f} GB ({util}%)"
                    )


# ------------------------------------------------------------
# 7. OPTIMIZED Training Arguments (FIXED)
# ------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./qwen3-0.6b-optimized",
    overwrite_output_dir=True,
    # 1. FIX FOR TORCH.COMPILE ERROR:
    remove_unused_columns=False,
    # 2. Batch Size Strategy for A4500 (20GB)
    per_device_train_batch_size=12,
    gradient_accumulation_steps=4,
    # 3. Optimization flags
    gradient_checkpointing=True,
    bf16=True,
    tf32=True,
    # 4. Data Loading
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    # 5. Training config
    max_steps=300,
    learning_rate=5e-4,
    lr_scheduler_type="cosine",
    warmup_steps=30,
    optim="adamw_torch_fused",
    # 6. Logging
    logging_steps=10,
    report_to="tensorboard",
    logging_dir=log_directory,
    save_strategy="no",
)

# ------------------------------------------------------------
# 8. Trainer & Train
# ------------------------------------------------------------
performance_callback = PerformanceCallback(max_seq_len=MAX_SEQ_LEN)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=packed_dataset,
    data_collator=data_collator,
    callbacks=[performance_callback],
)

print("\nStarting training (Initial step will be slow due to compilation)...")
trainer.train()

trainer.save_model("./qwen3-0.6b-final-optimized")
print("Done.")
