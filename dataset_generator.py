import pandas as pd
import random
import string

INPUT_CSV = "tinystories.csv"
OUTPUT_CSV = "tinystories_modified.csv"
HASHES_CSV = "hashes.csv"

TEXT_COLUMN = "text"

HASH_LENGTH = 16        
HASHES_PER_TEXT = 3

def generate_hash(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


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
    row_hashes = [generate_hash(HASH_LENGTH) for _ in range(HASHES_PER_TEXT)]
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