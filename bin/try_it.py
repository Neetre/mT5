from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Neetree/KoLama",
    max_seq_length = 128,
    dtype = None,
    load_in_4bit = True,
    token = "<your_token>", # Replace with your HF token, or use the HF_TOKEN environment variable
)

FastLanguageModel.for_inference(model)
inputs = tokenizer('''Translate from English to Korean: Hello!
''', return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)