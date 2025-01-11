from huggingface_hub import login
from datasets import load_dataset

# Login to the Hugging Face Hub
HUGGINGFACE_TOKEN = input("Inserisci il tuo token huggingface: ")
login(HUGGINGFACE_TOKEN)

INSTRUCTION = "Translate the following text from English to Korean:"

def combine_texts(example):
    """
    Combines texts in LLaMA chat format for fine-tuning.
    Args:
        example (dict): A single example from the dataset.

    Returns:
        dict: A dictionary containing the formatted 'text' key.
    """
    instruction = INSTRUCTION
    prompt = example['translation']['en']
    answer = example['translation']['ko']
    combined = {
        "text": f"""
{instruction}

### Prompt
{prompt}

### Answer
{answer}
"""
    }
    return combined

def clean_test(example):
    """
    Cleans the test dataset by removing the 'Answer' content.
    Args:
        example (dict): A single example from the test dataset.

    Returns:
        dict: A modified example with an empty 'Answer' section.
    """
    text = example['text']
    # Split and remove the Answer part
    text = text.split("### Answer")[0]
    text += "### Answer\n"  # Add an empty answer section
    return {"text": text}

def create_datasets():
    dataset = load_dataset("Neetree/raw_enko_opus_CCM")  # Structure: {'id': '0', 'translation': {'en': '...', 'ko': '...'}}
    dataset = dataset["train"].train_test_split(test_size=0.2)

    finetuning_dataset = dataset['train'].map(combine_texts, remove_columns=["id", "translation"])
    test_dataset = dataset['test'].map(combine_texts, remove_columns=["id", "translation"])

    test_dataset = test_dataset.map(clean_test)

    train_dataset = finetuning_dataset.train_test_split(test_size=0.1)['train']
    test_dataset = finetuning_dataset.train_test_split(test_size=0.1)['test']  # Cleaned dataset already processed

    print(train_dataset)
    print(test_dataset)

    return train_dataset, test_dataset
