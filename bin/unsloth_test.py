'''
Video link: https://www.youtube.com/watch?v=3eq84KrdTWY


da modificare per adattarlo al nostro caso

'''

from unsloth import FastLanguageModel
import torch
from prep_dataset import create_datasets
from evaluate import load


HUGGINGFACE_TOKEN = input("Inserisci il tuo token huggingface: ")

max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally! Suggested 4096, 8192, 16384
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


train_dataset, test_dataset = create_datasets()


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 4,
    packing = True, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size=64,
        gradient_accumulation_steps=32,
        warmup_steps=500,
        num_train_epochs=3,  # Train for 3 epochs
        learning_rate=3e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

trainer_stats = trainer.train() # Train the model!

bleu = load("bleu")

def evaluate_model(model, tokenizer, test_dataset):
    predictions = []
    references = []
    for example in test_dataset:
        inputs = tokenizer(example["text"], return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=512)
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        predictions.append(prediction)
        references.append(example["text"].split("=>")[1].strip())
    return bleu.compute(predictions=predictions, references=references)

bleu_score = evaluate_model(model, tokenizer, test_dataset)
print(f"BLEU Score: {bleu_score}")


model.save_pretrained_merged("KoLama", tokenizer, save_method="merged_16bit")
model.push_to_hub_merged("Neetree/KoLama", tokenizer, save_method="merged_16bit", token=HUGGINGFACE_TOKEN)