from huggingface_hub import login
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from optimization import setup_training_environment, optimize_model_memory, GPU_CONFIGS, DatasetOptimizer
import os
import evaluate
import numpy as np
load_dotenv()

HF_KEY = os.getenv("HF_KEY")
login("") # HF_KEY 

gpu_type = "A100_80"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

dataset = load_dataset("Neetree/raw_enko_opus_CCM")
dataset = dataset["train"].train_test_split(test_size=0.2)
print(dataset)
# new_train_size = len(dataset["train"]) - 24000
# dataset["train"] = dataset["train"].select(range(new_train_size))
# print(dataset)

checkpoint = "google/mt5-base"
processor = AutoTokenizer.from_pretrained(checkpoint)

dataset_optimizer = DatasetOptimizer(GPU_CONFIGS[gpu_type])
optimized_dataset = dataset_optimizer.optimize_dataset(dataset, processor)

source_lang = "en"
target_lang = "ko"
# prefix = "translate English to Korean: "

def preprocess_function(examples):
    inputs = [example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]

    model_inputs = processor(
        inputs,
        text_target=targets,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    input_lengths = [len(processor.tokenize(text)) for text in inputs]
    target_lengths = [len(processor.tokenize(text)) for text in targets]
    print(f"Input length stats: min={min(input_lengths)}, max={max(input_lengths)}, avg={sum(input_lengths)/len(input_lengths)}")
    print(f"Target length stats: min={min(target_lengths)}, max={max(target_lengths)}, avg={sum(target_lengths)/len(target_lengths)}")
    
    return model_inputs

tokenized_dataset = optimized_dataset.map(preprocess_function, batched=True, remove_columns=["id", "length", "translation"])

def verify_dataset(dataset, processor):
    for i in range(3):
        example = dataset[i]
        input_ids = np.array(example['input_ids'])
        labels = np.array(example['labels'])

        print(f"\nExample {i+1}:")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Decoded input: {processor.decode(input_ids)}")
        print(f"Decoded label: {processor.decode([l for l in labels if l != -100])}")

        print(f"Number of non-padding input tokens: {(input_ids != processor.pad_token_id).sum()}")
        print(f"Number of non-padding label tokens: {(labels != -100).sum()}")

# verify_dataset(tokenized_dataset["train"], processor)


def check_tokenization(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    print(f"Original text: {text}")
    print(f"Tokenized: {tokens}")
    decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))
    print(f"Decoded back: {decoded}")
    print(f"Number of tokens: {len(tokens)}")

# check_tokenization("Hello, how are you?", processor)


data_collator = DataCollatorForSeq2Seq(tokenizer=processor, model=checkpoint)

metric = evaluate.load("sacrebleu") # sacrebleu for BERTscore 

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.where((preds >= 0) & (preds < processor.vocab_size), preds, processor.unk_token_id)
    decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, processor.pad_token_id)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    metric = evaluate.load("sacrebleu")
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {"bleu": result["score"]}

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

training_args, memory_tracker  = setup_training_environment("koen_mT5", gpu_type)  # 4090, A100_40, A100_80, H100
model = optimize_model_memory(model, GPU_CONFIGS[gpu_type])

def check_model_updates(model, inputs):
    initial_params = {name: param.clone() for name, param in model.named_parameters()}

    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()

    for name, param in model.named_parameters():
        initial_param = initial_params[name]
        if param.grad is None:
            print(f"No gradient for {name}")
        else:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            initial_param_norm = initial_param.norm().item()
            param_change_norm = (param - initial_param).norm().item()
            print(f"{name}: grad_norm={grad_norm:.6f}, param_norm={param_norm:.6f}, initial_param_norm={initial_param_norm:.6f}, param_change_norm={param_change_norm:.6f}")


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, memory_tracker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(memory_tracker)
        
trainer = CustomTrainer(
    memory_tracker=memory_tracker,
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("Verifying training data...")
verify_dataset(tokenized_dataset["train"], processor)

print("\nChecking initial model state...")
sample_batch = next(iter(trainer.get_train_dataloader()))
check_model_updates(model, sample_batch)

try:
    print("Training model...")
    trainer.train()
except KeyboardInterrupt:
    print("Training interrupted")
except Exception as e:
    print(f"Training failed: {e}")
    model.save_pretrained("koen_mT5")

print(f"Peak memory usage: {memory_tracker.peak_memory:.2f} GB")
print("\nMemory trace:")
for checkpoint in memory_tracker.memory_trace:
    print(f"Time: {checkpoint['time']:.2f}s, GPU Memory: {checkpoint['gpu_allocated']:.2f} GB")

trainer.push_to_hub()


def validate_translations(model, tokenizer, test_samples=5):
    """Test the model on a few examples during training"""
    test_texts = [
        "Hello, how are you?",
        "The weather is beautiful today.",
        "I love learning new languages.",
        "This is a complex sentence with multiple clauses and technical terms.",
        "Please translate this sentence carefully."
    ]

    for text in test_texts:
        input_text = f"translate English to Korean: {text}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
        outputs = model.generate(
            inputs.input_ids.to(model.device),
            max_length=128,
            num_beams=5,
            length_penalty=0.6,
            early_stopping=True
        )
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nInput: {text}")
        print(f"Translation: {translated}")


def translate_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model.generate(
        inputs.input_ids,
        max_length=128,
        num_beams=5,
        length_penalty=1.0,
        early_stopping=True,
        do_sample=False,
        temperature=0.7,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


text = "translate English to Korean: Legumes share resources with nitrogen-fixing bacteria."

from transformers import pipeline

translator = pipeline("translation_english_to_korean", model="Neetree/koen_mT5")
translator(text)
'''
processor = AutoTokenizer.from_pretrained("Neetree/koen_mT5")
inputs = processor(text, return_tensors="pt").input_ids

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("Neetree/koen_mT5")
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)

processor.decode(outputs[0], skip_special_tokens=True)
'''