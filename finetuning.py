import pandas as pd
DATA_PATH = "/content/drive/MyDrive/aisec/tinystories_modified.csv"


df = pd.read_csv(DATA_PATH)

print("Columns in the DataFrame:")
print(df.columns.tolist())

print("\nFirst 10 rows of the DataFrame:")
display(df.head(10))

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments
)


MODEL_NAME = "distilgpt2"
TEXT_COL = "text"
OUTPUT_DIR = "./finetuned_llm_hash_extraction"

EPOCHS = 5
BATCH_SIZE = 4
MAX_LENGTH = 512
LEARNING_RATE = 5e-5


df = pd.read_csv(DATA_PATH)
texts = df[TEXT_COL].astype(str).tolist()

print(f"Total training samples: {len(texts)}")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))


class LMDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx]}

dataset = LMDataset(texts)

def collate_fn(batch):
    texts = [item["text"] for item in batch]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    encodings["labels"] = encodings["input_ids"].clone()
    return encodings


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=100,
    weight_decay=0.01,

    remove_unused_columns=False,

    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,

    gradient_accumulation_steps=2,
    fp16=torch.cuda.is_available(),

    report_to="none",
    dataloader_num_workers=2,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)


print("\nStarting fine-tuning...")
trainer.train()


trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n Fine-tuning finished!")
print(f" Model saved to: {OUTPUT_DIR}")

# =====================
# QUICK TEST
# =====================
print("\n" + "="*50)
print("QUICK TEST")
print("="*50)

model.eval()
test_prompt = texts[0][:50]

print(f"\nPrompt: {test_prompt}...")

inputs = tokenizer(test_prompt, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=MAX_LENGTH,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated:\n{generated}")

# dopisac pomiedzy krokami metryki i w miedzy czasie cos pormpotowac