import pandas as pd
import random
import string
import hashlib
import os

INPUT_CSV = "tinystories.csv"
OUTPUT_CSV = "tinystories_modified.csv"
HASHES_CSV = "hashes.csv"

TEXT_COLUMN = "text"

HASH_LENGTH = None  # Niepotrzebne
HASHES_PER_TEXT = 1 

def generate_short_hash_with_checksum():
    # 32 bity → 8 znaków w hex
    core = os.urandom(4).hex()

    # checksum = SHA256(core), pierwszy bajt (2 znaki w hex)
    checksum = hashlib.sha256(core.encode()).hexdigest()[:2]

    return f"{core}-{checksum}"

def insert_hashes_at_word_boundaries(text, hash_list):
    words = text.split()

    if not words:
        return text + " " + " ".join(hash_list)

    positions = sorted(
        random.sample(range(len(words) + 1), k=min(len(hash_list), len(words) + 1)),
        reverse=True
    )

    for pos, hash_str in zip(positions, hash_list):
        words.insert(pos, hash_str)

    return " ".join(words)

df = pd.read_csv(INPUT_CSV)
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

all_hashes = []
hashes_per_row = []

for _ in range(len(df)):
    row_hashes = [
        generate_short_hash_with_checksum()
        for _ in range(HASHES_PER_TEXT)
    ]
    hashes_per_row.append(row_hashes)
    all_hashes.extend(row_hashes)

df[TEXT_COLUMN] = [
    insert_hashes_at_word_boundaries(text, row_hashes)
    for text, row_hashes in zip(df[TEXT_COLUMN], hashes_per_row)
]

df.to_csv(OUTPUT_CSV, index=False)

hash_df = pd.DataFrame({
    "hash": all_hashes
})

hash_df.to_csv(HASHES_CSV, index=False)
