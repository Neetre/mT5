import multiprocessing as mp
from tqdm import tqdm
from typing import Dict, Tuple, List, Iterator
import requests
import json
import gzip
from zipfile import ZipFile
from itertools import islice
from contextlib import contextmanager
import os
import argparse
import warnings
import logging
import transformers
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

from transformers import MT5Tokenizer

corpus = "CCMatrix"
src_lang = "en"
trc_lang = "ko"
src_full = "English"
trg_full = "Korean"

# Argument parsing
parser = argparse.ArgumentParser(description="Download and preprocess OPUS data")
parser.add_argument("-d", "--data_dir", type=str, default="../data/opus", help="Directory to save the data")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
parser.add_argument("-n", "--max_pairs", type=int, default=10**7, help="Maximum number of sentence pairs to process")
parser.add_argument("-b", "--batch_size", type=int, default=1000, help="Number of sentences to process in each batch")
parser.add_argument("-m", "--max_seq_len", type=int, default=128, help="Maximum sequence length")
parser.add_argument("--num_workers", type=int, default=mp.cpu_count(), help="Number of workers for multiprocessing")
args = parser.parse_args()

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', "opus", f"{src_lang}-{trc_lang}")
os.makedirs(DATA_ROOT, exist_ok=True)

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

class DatasetIterator:
    def __init__(self, data_dir: str, batch_size: int):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.file_pairs = self._get_file_pairs()
        self.total_pairs = self._count_total_pairs()
    
    def _get_file_pairs(self) -> List[Tuple[str, str]]:
        """Get pairs of English and Korean files."""
        files = os.listdir(self.data_dir)
        src_files = sorted([f for f in files if f.endswith(f'.{src_lang}')])
        trg_files = sorted([f for f in files if f.endswith(f'.{trc_lang}')])
        if len(src_files) != len(trg_files):
            raise ValueError("Mismatch in the number of English and Korean files.")
        return list(zip(src_files, trg_files))

    def _count_total_pairs(self) -> int:
        """Count total number of pairs across all files."""
        total = 0
        for src_file, trg_files in self.file_pairs:
            with self._open_file_pair(src_file, trg_files) as (src_f, trg_f):
                total += sum(1 for _ in src_f)
        return total

    @contextmanager
    def _open_file_pair(self, src_file: str, trg_file: str):
        """Context manager to open file pairs."""
        try:
            with open(os.path.join(self.data_dir, src_file), 'r', encoding='utf-8') as src_f, \
                 open(os.path.join(self.data_dir, trg_file), 'r', encoding='utf-8') as trg_f:
                yield src_f, trg_f
        except Exception as e:
            logger.error(f"Error opening file pair {src_file}, {trg_file}: {e}")
            raise

    def __iter__(self) -> Iterator[List[Dict[str, str]]]:
        """Iterate over batches of sentence pairs."""
        for src_file, trg_file in self.file_pairs:
            with self._open_file_pair(src_file, trg_file) as (src_f, trg_f):
                while True:
                    src_lines = list(islice(src_f, self.batch_size))
                    trg_lines = list(islice(trg_f, self.batch_size))

                    if not src_lines or not trg_lines:
                        break

                    if len(trg_lines) != len(trg_lines):
                        logger.warning(f"Mismatch in the number of lines in {src_file} and {trg_file}")
                        continue

                    batch = [{"en": en_line.strip(), "ko": ko_line.strip()} for en_line, ko_line in zip(src_lines, trg_lines)]
                    yield batch

def download_file(url, output_path):
    """Download a file with a progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as file, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
    except Exception as e:
        logger.error(f"Error downloading file {url}: {e}")
        raise

def download_opus_data(corpus, source_lang, target_lang):
    """Download OPUS data for the specified corpus and languages."""
    api_url = f"http://opus.nlpl.eu/opusapi/?corpus={corpus}&source={source_lang}&target={target_lang}&preprocessing=moses&version=latest"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = json.loads(response.text)

        os.makedirs(DATA_ROOT, exist_ok=True)

        for item in data["corpora"]:
            url = item["url"]
            filename = os.path.basename(url)
            output_path = os.path.join(DATA_ROOT, filename)

            logger.info(f"Downloading {filename}...")
            download_file(url, output_path)

            if filename.endswith('.xml.gz'):
                logger.info(f"Extracting {filename}...")
                with gzip.open(output_path, 'rb') as f_in:
                    with open(output_path[:-3], 'wb') as f_out:
                        f_out.write(f_in.read())
            elif filename.endswith('.zip'):
                logger.info(f"Extracting {filename}...")
                with ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(DATA_ROOT)

            logger.info(f"Successfully processed {filename}")
    except Exception as e:
        logger.error(f"Error downloading OPUS data: {e}")
        raise


def get_data_dir(output_dir, split):
    """Create and return language/split specific directory"""
    data_dir = os.path.join(output_dir, split)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

prefix = f"translate {src_full} to {trg_full}: "

def process_batch_worker(batch: List[Dict[str, str]], max_seq_len: int) -> Dict[str, List]:
    """Tokenize a batch of sentences using the mT5 tokenizer."""
    try:
        input_ids = []
        attention_masks = []
        labels = []
        src_texts = []
        trg_texts = []
        for item in batch:
            if not item.get(src_lang) or not item.get(trc_lang):
                continue
            inputs = prefix + item[src_lang]
            try:
                en_tokenized = tokenizer(
                    inputs,
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_len,
                    return_tensors="pt",
                )
                ko_tokenized = tokenizer(
                    item[trc_lang],
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_len,
                    return_tensors="pt",
                )

                input_ids.append(en_tokenized["input_ids"].squeeze().tolist())
                attention_masks.append(en_tokenized["attention_mask"].squeeze().tolist())
                labels.append(ko_tokenized["input_ids"].squeeze().tolist())
                src_texts.append(inputs)
                trg_texts.append(item[trc_lang])

            except Exception as e:
                logger.warning(f"Error processing item: {e}")
                continue

        return {
            "id": "dummy_id",
            "translation": {
                f"{src_lang}": src_texts,
                f"{trc_lang}": trg_texts
            },
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            
        }
    except Exception as e:
        logger.error(f"Worker process error: {e}")
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            f"{src_lang}_texts": [],
            f"{trc_lang}_texts": []
        }

def save_shard(shard: Dict[str, List], output_dir: str, shard_id: int, split: str):
    """Save a shard of the dataset to disk."""
    data_dir = get_data_dir(output_dir, split)
    try:
        if not shard["input_ids"]:
            logger.warning(f"Skipping empty shard {shard_id}")
            return

        dataset = Dataset.from_dict(shard)
        dataset.save_to_disk(os.path.join(data_dir, f"shard_{shard_id}.arrow"))
        logger.info(f"Saved shard {shard_id} to disk")
    except Exception as e:
        logger.error(f"Error saving shard {shard_id} to disk: {e}")
        raise


def preprocess_dataset(data_dir: str, output_dir: str, args):
    """Preprocess the dataset using multiprocessing and token-count-based sharding."""
    os.makedirs(output_dir, exist_ok=True)

    current_shard = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        f"{src_lang}_texts": [],
        f"{trc_lang}_texts": []
    }
    token_count = 0
    shard_id = 0
    num_pairs_processed = 0
    
    dataset_iterator = DatasetIterator(data_dir, args.batch_size)
    total_iterations = min(dataset_iterator.total_pairs // args.batch_size, args.max_pairs // args.batch_size)

    FUTURE_TIMEOUT = 10

    pool = mp.Pool(args.num_workers)
    batch = []
    futures = []

    try:
        pbar = tqdm(total=total_iterations, desc="Processing documents")
        
        for doc_batch in dataset_iterator:
            if num_pairs_processed >= args.max_pairs:
                break

            batch.extend(doc_batch)

            if len(batch) >= args.batch_size:
                current_batch = batch[:args.batch_size]
                batch = batch[args.batch_size:]
                futures.append(pool.apply_async(process_batch_worker, (current_batch, args.max_seq_len)))

            completed_futures = [f for f in futures if f.ready()]
            for future in completed_futures:
                futures.remove(future)
                try:
                    result = future.get(timeout=FUTURE_TIMEOUT)
                    if any(len(v) > 0 for v in result.values()):
                        for key in current_shard:
                            current_shard[key].extend(result[key])
                        token_count += sum(len(ids) for ids in result["input_ids"])
                        num_pairs_processed += len(result["input_ids"])
                        pbar.update(len(result["input_ids"]) // args.batch_size)

                        if token_count >= args.shard_size:
                            split = "val" if shard_id == 0 else "train"
                            save_shard(current_shard, output_dir, shard_id, split)
                            shard_id += 1
                            current_shard = {
                                "input_ids": [],
                                "attention_mask": [],
                                "labels": [],
                                f"{src_lang}_texts": [],
                                f"{trc_lang}_texts": []
                            }
                            token_count = 0
                except mp.TimeoutError:
                    logger.warning(f"Future timed out after {FUTURE_TIMEOUT} seconds, skipping...")
                    continue
                except Exception as e:
                    logger.error(f"Error processing future: {e}")
                    continue

        if batch:
            try:
                result = process_batch_worker(batch, args.max_seq_len)
                if any(len(v) > 0 for v in result.values()):
                    for key in current_shard:
                        current_shard[key].extend(result[key])
            except Exception as e:
                logger.error(f"Error processing final batch: {e}")

        logger.info("Processing remaining futures...")
        remaining_futures = len(futures)
        if remaining_futures > 0:
            for i, future in enumerate(futures, 1):
                try:
                    logger.info(f"Processing remaining future {i}/{remaining_futures}")
                    result = future.get(timeout=FUTURE_TIMEOUT)
                    if any(len(v) > 0 for v in result.values()):
                        for key in current_shard:
                            current_shard[key].extend(result[key])
                except mp.TimeoutError:
                    logger.warning(f"Future {i}/{remaining_futures} timed out, skipping...")
                    continue
                except Exception as e:
                    logger.error(f"Error processing remaining future {i}/{remaining_futures}: {e}")
                    continue

        if any(len(v) > 0 for v in current_shard.values()):
            split = "val" if shard_id == 0 else "train"
            save_shard(current_shard, output_dir, shard_id, split)

        pbar.close()
        logger.info("Dataset preprocessing completed successfully")

    except Exception as e:
        logger.error(f"Error in main processing loop: {e}")
        raise
    finally:
        logger.info("Cleaning up multiprocessing resources...")
        pool.close()
        pool.join()
        logger.info("Cleanup completed")

def main():
    if not os.path.exists(os.path.join(DATA_ROOT, f"{src_lang}-{trc_lang}.txt.zip")):
        try:
            download_opus_data(corpus=corpus, source_lang=src_lang, target_lang=trc_lang)
        except Exception as e:
            logger.error(f"Failed to download OPUS data: {e}")
            return
    temp_dir = os.path.join(DATA_ROOT, "temp")
    logger.info(f"Preprocessing dataset and saving to {temp_dir}...")

    try:
        preprocess_dataset(args.data_dir, temp_dir, args)

    except Exception as e:
        logger.error(f"Failed to preprocess dataset: {e}")
        return

if __name__ == "__main__":
    main()
