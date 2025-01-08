from huggingface_hub import login
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
import os
import evaluate
import numpy as np
load_dotenv()

HF_KEY = os.getenv("HF_KEY")
login() # HF_KEY 

dataset = load_dataset("Neetree/raw_enko_opus_CCM")
dataset = dataset["train"].train_test_split(test_size=0.2)
print(dataset)

checkpoint = "google/mt5-base"
processor = AutoTokenizer.from_pretrained(checkpoint)

source_lang = "en"
target_lang = "ko"
prefix = "translate English to Korean: "

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]

    model_inputs = processor(
        inputs,
        text_target=targets,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_lengths = [len(processor.tokenize(text)) for text in inputs]
    target_lengths = [len(processor.tokenize(text)) for text in targets]
    print(f"Input length stats: min={min(input_lengths)}, max={max(input_lengths)}, avg={sum(input_lengths)/len(input_lengths)}")
    print(f"Target length stats: min={min(target_lengths)}, max={max(target_lengths)}, avg={sum(target_lengths)/len(target_lengths)}")
    
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

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

verify_dataset(tokenized_dataset["train"], processor)


def check_tokenization(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    print(f"Original text: {text}")
    print(f"Tokenized: {tokens}")
    decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))
    print(f"Decoded back: {decoded}")
    print(f"Number of tokens: {len(tokens)}")

check_tokenization("Hello, how are you?", processor)


data_collator = DataCollatorForSeq2Seq(tokenizer=processor, model=checkpoint)

metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, processor.pad_token_id)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != processor.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

    result = {k: round(v, 4) for k, v in result.items()}
    return result

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir="koen_mT5",
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=False,  # Disable fp16 for debugging
    logging_dir="./logs",
    logging_steps=10,
    report_to=["tensorboard"],
    gradient_checkpointing=True,
    debug=True,
)

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


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        print(f"Current loss: {loss.item()}")
        return (loss, outputs) if return_outputs else loss


trainer = CustomSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Verifying training data...")
verify_dataset(tokenized_dataset["train"], processor)

print("\nChecking initial model state...")
sample_batch = next(iter(trainer.get_train_dataloader()))
check_model_updates(model, sample_batch)

trainer.train()

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
            max_length=256,
            num_beams=5,
            length_penalty=0.6,
            early_stopping=True
        )
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nInput: {text}")
        print(f"Translation: {translated}")


def translate_text(text, model, tokenizer):
    input_text = f"translate English to Korean: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
    outputs = model.generate(
        inputs.input_ids,
        max_length=256,
        num_beams=5,
        length_penalty=0.6,
        early_stopping=True,
        do_sample=False,
        temperature=1.0,
        no_repeat_ngram_size=3
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