import argparse
import os
import logging
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from huggingface_hub import login

from dotenv import load_dotenv
load_dotenv()
HF_KEY = os.getenv("HF_KEY")

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Upload preprocessed dataset to Hugging Face Hub")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing the preprocessed dataset shards",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        required=True,
        help="Hugging Face repository to which the dataset will be pushed",
    )
    return parser.parse_args()

def push_to_huggingface(output_dir: str, hf_repo: str):
    """Push the preprocessed dataset to Hugging Face Hub."""
    try:
        splits = ["train", "val"]
        combined_datasets = {}
        
        for split in splits:
            split_path = os.path.join(output_dir, split)
            if not os.path.exists(split_path):
                logger.warning(f"Split path {split_path} does not exist, skipping...")
                continue

            datasets = []
            shard_files = os.listdir(split_path)
            
            for shard_file in tqdm(shard_files, desc=f"Loading {split} shards"):
                try:
                    dataset = Dataset.load_from_disk(os.path.join(split_path, shard_file))
                    datasets.append(dataset)
                except Exception as e:
                    logger.error(f"Error loading shard {shard_file}: {e}")
                    continue
            
            if datasets:
                combined_datasets[split] = concatenate_datasets(datasets)
                logger.info(f"Successfully combined {len(datasets)} shards for {split} split")
            else:
                logger.warning(f"No valid shards found for {split} split")
        
        if not combined_datasets:
            raise ValueError("No valid datasets found to push to Hub")
            
        for split, dataset in combined_datasets.items():
            dataset.push_to_hub(hf_repo, split=split, private=True)
            logger.info(f"Successfully pushed {split} split to {hf_repo}")
            
    except Exception as e:
        logger.error(f"Error pushing dataset to Hugging Face Hub: {e}")
        raise
    

def main():
    args = parse_args()
    try:
        logger.info("Logging in to Hugging Face Hub...")
        login(token=HF_KEY)

        if not os.path.exists(args.output_dir):
            raise FileNotFoundError(f"Directory {args.output_dir} not found")
        if not os.path.isdir(args.output_dir):
            raise NotADirectoryError(f"{args.output_dir} is not a directory")
        if not os.listdir(args.output_dir):
            raise ValueError(f"Directory {args.output_dir} is empty")
        if not args.hf_repo:
            raise ValueError("Please provide a Hugging Face repository name")

        logger.info("Pushing preprocessed dataset to Hugging Face Hub...")
        push_to_huggingface(args.output_dir, args.hf_repo)
    except Exception as e:
        logger.error(f"Error pushing dataset to Hugging Face Hub: {e}")
        raise


if __name__ == "__main__":
    main()
