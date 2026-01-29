import multiprocessing
import os
import regex as re
from collections import defaultdict


def read_file(input_path: str | os.PathLike):
    with open(input_path, "r") as f:
        return f.read()

def split_text_by_special_tokens(text: str, special_tokens: list[str]):
    # Use re.split with "|".join(special_tokens) as the separator
    pattern = "|".join(map(re.escape, special_tokens))
    chunks = re.split(pattern, text)
    return [chunk for chunk in chunks if chunk]

def word2bytes(word: str)-> tuple[int, ...]:
    word_encoded = word.encode("utf-8")
    return tuple(list(word_encoded))

def pre_tokenize_chunk(chunk: str)-> dict[tuple[int, ...], int]:
    word_count = defaultdict(int)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for match in re.finditer(PAT, chunk):
        word = match.group(0)
        word_bytes = word2bytes(word)
        if len(word_bytes) > 1:
            word_count[word_bytes] += 1
    return word_count

def merge_dicts(dicts: list[dict[tuple[int, ...], int]])-> dict[tuple[int, ...], int]:
    merged_dict = defaultdict(int)
    for dict in dicts:
        for key, value in dict.items():
            merged_dict[key] += value
    return merged_dict

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]):
    text = read_file(input_path)

    # Split text into chunks with special_tokens
    chunks = split_text_by_special_tokens(text, special_tokens)

    # Parallelizing pre-tokenization
    ## Get multi counts using multiprocessing
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        word_counts = pool.map(pre_tokenize_chunk, chunks)
    ## Merge word counts
    merged_word_count = merge_dicts(word_counts)