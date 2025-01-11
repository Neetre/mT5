from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import login
import os
from unsloth import FastLanguageModel

token = input("Insert your huggingface token: ")
login(token)

# output_dir = ""
# checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
# latest_checkpoint = os.path.join(output_dir, sorted(checkpoint_dirs)[-1])

latest_checkpoint = "KoLama"
model = AutoModelForSeq2SeqLM.from_pretrained(latest_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)
model = FastLanguageModel
repo_name = "Neetree/KoLama"
if True: model.save_pretrained_merged("KoLama", tokenizer, save_method = "merged_16bit",)
if True: model.push_to_hub_merged("Neetree/KoLama", tokenizer, save_method = "merged_16bit", token = token)

print(f"Model and tokenizer pushed to {repo_name} from checkpoint {latest_checkpoint}")