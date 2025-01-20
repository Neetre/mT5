from huggingface_hub import login
from datasets import load_dataset

HUGGINGFACE_TOKEN = input("Insert your Huggingface token: ")
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
    return f"Translate English to Korean: {example['english']} => {example['korean']}"

def clean_test(example):
    """
    Cleans the test dataset by removing the 'Answer' content.
    Args:
        example (dict): A single example from the test dataset.

    Returns:
        dict: A modified example with an empty 'Answer' section.
    """
    text = example['text']
    text = text.split(" => ")[0]
    text += " => \n"
    return {"text": text}

def create_datasets():
    dataset = load_dataset("Neetree/raw_enko_opus_CCM")  # Structure: {'id': '0', 'translation': {'en': '...', 'ko': '...'}}
    dataset = dataset["train"].train_test_split(test_size=0.2)

    train_dataset = dataset['train'].map(combine_texts, remove_columns=["id", "translation"])
    test_dataset = dataset['test'].map(combine_texts, remove_columns=["id", "translation"])

    test_dataset_no_asw = test_dataset.map(clean_test)

    print(train_dataset)
    print(test_dataset)
    print(test_dataset[0])

    return train_dataset, test_dataset, test_dataset_no_asw


train_dataset, test_dataset = create_datasets()