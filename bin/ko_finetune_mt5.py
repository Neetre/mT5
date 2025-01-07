from huggingface_hub import login
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoProcessor
from transformers import DataCollatorForSeq2Seq
import os
import evaluate
import numpy as np
load_dotenv()

HF_KEY = os.getenv("HF_KEY")
login() # HF_KEY 

from datasets import load_dataset
dataset = load_dataset("Neetree/raw_enko_opus_CCM")
dataset = dataset["train"].train_test_split(test_size=0.2)
print(dataset)

checkpoint = "google/mt5-base"
processor = AutoProcessor.from_pretrained(checkpoint)

source_lang = "en"
target_lang = "ko"
prefix = "translate English to Korean: "

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = processor(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
train_dataset = tokenized_dataset["train"]
val_dataset = tokenized_dataset["test"]

data_collator = DataCollatorForSeq2Seq(processor=processor, model=checkpoint)

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

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != processor.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir="koen_mT5",
    eval_strategy="steps",
    eval_steps=1000,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    save_steps=1000,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    gradient_accumulation_steps=4,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True, #change to bf16=True for XPU
    push_to_hub=True,
    logging_dir="../logs",
    logging_steps=100,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub()

text = "translate English to Korean: Legumes share resources with nitrogen-fixing bacteria."

from transformers import pipeline

translator = pipeline("translation_english_to_korean", model="Neetree/koen_mT5")
translator(text)

processor = AutoTokenizer.from_pretrained("Neetree/koen_mT5")
inputs = processor(text, return_tensors="pt").input_ids

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("Neetree/koen_mT5")
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)

processor.decode(outputs[0], skip_special_tokens=True)