import torch
import psutil
import os
from dataclasses import dataclass
from transformers import TrainerCallback
import gc
import time
from transformers import Seq2SeqTrainingArguments

@dataclass
class GPUConfig:
    """Configuration for different GPU types"""
    name: str
    memory_gb: int
    batch_size: int
    gradient_accumulation: int
    fp16: bool
    bf16: bool
    gradient_checkpointing: bool
    max_length: int
    optimal_learning_rate: float

GPU_CONFIGS = {
    "4090": GPUConfig(
        name="RTX 4090",
        memory_gb=24,
        batch_size=24,
        gradient_accumulation=4,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        max_length=128,
        optimal_learning_rate=3e-4
    ),
    "A100_40": GPUConfig(
        name="A100 40GB",
        memory_gb=40,
        batch_size=48,
        gradient_accumulation=2,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        max_length=256,
        optimal_learning_rate=3e-4
    ),
    "A100_80": GPUConfig(
        name="A100 80GB",
        memory_gb=80,
        batch_size=96,
        gradient_accumulation=1,
        fp16=False,
        bf16=True,
        gradient_checkpointing=False,
        max_length=512,
        optimal_learning_rate=4e-4
    ),
    "H100": GPUConfig(
        name="H100",
        memory_gb=80,
        batch_size=128,
        gradient_accumulation=1,
        fp16=False,
        bf16=True,
        gradient_checkpointing=False,
        max_length=512,
        optimal_learning_rate=5e-4
    )
}

class MemoryTracker(TrainerCallback):
    """Tracks memory usage during training"""
    def __init__(self, gpu_config: GPUConfig):
        self.gpu_config = gpu_config
        self.memory_trace = []
        self.peak_memory = 0
        self.start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.log_memory_usage("Training started")
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            self.log_memory_usage(f"Step {state.global_step}")
            
    def log_memory_usage(self, checkpoint: str):
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_memory_cached = torch.cuda.memory_reserved() / 1e9
            
            self.peak_memory = max(self.peak_memory, gpu_memory_allocated)
            
            memory_info = {
                'checkpoint': checkpoint,
                'gpu_allocated': gpu_memory_allocated,
                'gpu_cached': gpu_memory_cached,
                'ram_used': psutil.Process(os.getpid()).memory_info().rss / 1e9,
                'time': time.time() - self.start_time if self.start_time else 0
            }
            
            self.memory_trace.append(memory_info)
            
            print(f"\n=== Memory Usage at {checkpoint} ===")
            print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} GB")
            print(f"GPU Memory Cached: {gpu_memory_cached:.2f} GB")
            print(f"RAM Used: {memory_info['ram_used']:.2f} GB")

            if gpu_memory_allocated > self.gpu_config.memory_gb * 0.85:
                print(f"⚠️ Warning: High GPU memory usage ({gpu_memory_allocated:.2f}/{self.gpu_config.memory_gb} GB)")

def optimize_model_memory(model, gpu_config: GPUConfig):
    """Apply memory optimizations based on GPU config"""

    if gpu_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if hasattr(model, "config"):
        if gpu_config.memory_gb < 40:
            model.config.use_cache = False

    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    
    return model

def find_optimal_batch_size(model, tokenizer, gpu_config: GPUConfig):
    """Find the optimal batch size through binary search"""
    
    def test_batch_size(batch_size: int) -> bool:
        try:
            torch.cuda.empty_cache()
            sample_input = "This is a test sentence" * 10
            inputs = tokenizer(
                [sample_input] * batch_size,
                return_tensors="pt",
                padding='max_length',
                max_length=gpu_config.max_length,
                truncation=True
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            del outputs, inputs
            torch.cuda.empty_cache()
            return True
        except RuntimeError:
            return False

    left, right = 1, gpu_config.batch_size * 2
    optimal_batch_size = 1
    
    while left <= right:
        mid = (left + right) // 2
        if test_batch_size(mid):
            optimal_batch_size = mid
            left = mid + 1
        else:
            right = mid - 1

    return int(optimal_batch_size * 0.9)

class DatasetOptimizer:
    """Optimizes dataset for efficient training"""
    
    def __init__(self, gpu_config: GPUConfig):
        self.gpu_config = gpu_config
        
    def optimize_dataset(self, dataset, tokenizer):
        """Apply dataset optimizations"""
        
        def filter_samples(example):
            src_len = len(tokenizer.encode(example['translation']['en']))
            tgt_len = len(tokenizer.encode(example['translation']['ko']))
            return src_len <= self.gpu_config.max_length and tgt_len <= self.gpu_config.max_length

        filtered_dataset = dataset.filter(filter_samples)

        def get_length(example):
            return len(tokenizer.encode(example['translation']['en']))
        
        sorted_dataset = filtered_dataset.map(
            lambda x: {'length': get_length(x)},
            num_proc=4
        ).sort('length')
        
        return sorted_dataset

def setup_training_environment(model_name: str, gpu_type: str):
    """Complete setup for training environment"""
    
    if gpu_type not in GPU_CONFIGS:
        raise ValueError(f"Unsupported GPU type. Choose from: {list(GPU_CONFIGS.keys())}")
    
    gpu_config = GPU_CONFIGS[gpu_type]

    memory_tracker = MemoryTracker(gpu_config)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{model_name}_{gpu_type}",
        learning_rate=gpu_config.optimal_learning_rate,
        per_device_train_batch_size=gpu_config.batch_size,
        per_device_eval_batch_size=gpu_config.batch_size,
        gradient_accumulation_steps=gpu_config.gradient_accumulation,
        fp16=gpu_config.fp16,
        bf16=gpu_config.bf16,
        gradient_checkpointing=gpu_config.gradient_checkpointing,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        max_grad_norm=1.0,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=2000,
        predict_with_generate=True,
        generation_max_length=gpu_config.max_length,
        optim="adamw_torch",
        group_by_length=True
    )
    
    return training_args, memory_tracker