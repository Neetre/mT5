from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import login
import os

login("")

output_dir = "koen_mT5"
checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
latest_checkpoint = os.path.join(output_dir, sorted(checkpoint_dirs)[-1])

model = AutoModelForSeq2SeqLM.from_pretrained(latest_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)

repo_name = "Neetree/koen_mT5"
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print(f"Model and tokenizer pushed to {repo_name} from checkpoint {latest_checkpoint}")