from contextlib import contextmanager
from itertools import islice
from typing import Dict, Iterator, List, Tuple
import gzip
import json
import logging
import os
from zipfile import ZipFile
import requests
from tqdm import tqdm
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


corpus = "CCMatrix"
src_lang = "en"
trc_lang = "ko"
src_full = "English"
trg_full = "Korean"

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', "opus", f"{src_lang}-{trc_lang}")
os.makedirs(DATA_ROOT, exist_ok=True)

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

                    if len(src_lines) != len(trg_lines):
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


def process_batch(batch: List[Dict[str, str]], num: int) -> Tuple[Dict[str, List], int]:
    """
    Process a batch of sentence pairs and return a dictionary shard.
    
    Args:
        batch: A list of dictionaries containing sentence pairs.
        num: The starting ID for the batch.
    
    Returns:
        A tuple containing:
        - A dictionary shard with keys "id" and "translation".
        - The next starting ID for the next batch.
    """
    shard = {
        "id": [],
        "translation": []
    }

    for idx, pair in enumerate(batch):
        shard["id"].append(f"{num + idx}")
        shard["translation"].append(pair)

    num += len(batch)

    return shard, num


def save_shard(shard: Dict[str, List], shard_id: int, split: str):
    """Save a shard of the dataset to disk."""
    data_dir = get_data_dir(DATA_ROOT, split)
    try:
        if not shard["id"]:
            logger.warning(f"Skipping empty shard {shard_id}")
            return

        dataset = Dataset.from_dict(shard)
        dataset.save_to_disk(os.path.join(data_dir, f"shard_{shard_id}.arrow"))
        logger.info(f"Saved shard {shard_id} to disk")
    except Exception as e:
        logger.error(f"Error saving shard {shard_id} to disk: {e}")
        raise


def main():
    if not os.path.exists(os.path.join(DATA_ROOT, f"{src_lang}-{trc_lang}.txt.zip")):
        try:
            logger.info("Downloading OPUS data...")
            download_opus_data(corpus=corpus, source_lang=src_lang, target_lang=trc_lang)
        except Exception as e:
            logger.error(f"Failed to download OPUS data: {e}")
            return

    try:
        logger.info("Processing OPUS data...")
        iterator = DatasetIterator(DATA_ROOT, batch_size=32)
        num = 0
        shard_id = 0
        current_shard = {
            "id": [],
            "translation": []
        }
        for batch in tqdm(iterator, total=iterator.total_pairs):
            batch, num = process_batch(batch, num)
            for key in current_shard:
                current_shard[key].extend(batch[key])

            if len(current_shard["id"]) >= 100000:
                split = "val" if shard_id == 0 else "train"
                save_shard(current_shard, shard_id, split)
                shard_id += 1
                current_shard = {
                    "id": [],
                    "translation": []
                }
        if current_shard:
            split = "val" if shard_id == 0 else "train"
            save_shard(current_shard, shard_id, split)
        
        logger.info("Finished processing OPUS data")
    except Exception as e:
        logger.error(f"Error processing OPUS data: {e}")
        raise


if __name__ == "__main__":
    main()
