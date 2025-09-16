import re
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# ---------- 1. Load CSV
data_path = r"C:/Users/explo/Desktop/Mental Health Chatbot/data/NLP mental health.csv"
df = pd.read_csv(data_path)

print("Raw data shape:", df.shape)
print(df.head())

# ---------- 2. Normalize & clean
def normalize_text(text: str) -> str:
    text = str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return text

df['context'] = df['context'].apply(normalize_text)
df['response'] = df['response'].apply(normalize_text)

# Remove too short/long
def response_length_ok(resp: str, min_len=4, max_len=200) -> bool:
    tokens = resp.split()
    return min_len <= len(tokens) <= max_len

df = df[df['response'].apply(response_length_ok)]

# Remove unsafe content
banned = ["reddit", "suicide", "kill myself", "self-harm", "years old"]
def is_safe(resp: str) -> bool:
    low = resp.lower()
    return not any(w in low for w in banned)

df = df[df['response'].apply(is_safe)]

print("Cleaned data shape:", df.shape)

# ---------- 3. Multi-turn stitching
conversations = []
history_window = 3  # keep last 3 exchanges

history = []
for _, row in df.iterrows():
    user_turn = f"<user>: {row['context']}"
    bot_turn = f"<bot>: {row['response']}"
    history.append(user_turn)
    history.append(bot_turn)

    if len(history) >= 2 * history_window:
        # take last N turns
        conv_context = " ".join(history[-(2*history_window):-1])  # all but last bot
        conv_response = history[-1]  # last bot
        conversations.append({"Context": conv_context, "Response": conv_response})

# DataFrame → HuggingFace dataset
hf_ds = Dataset.from_pandas(pd.DataFrame(conversations))
hf_ds = hf_ds.train_test_split(test_size=0.1)

# ---------- 4. Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

MAX_LEN = 256  # allow longer context

def encode(example):
    enc = tokenizer(
        example['Context'] + " " + example['Response'] + tokenizer.eos_token,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    return {
        "input_ids": enc['input_ids'],
        "attention_mask": enc['attention_mask'],
        "labels": enc['input_ids']
    }

encoded_ds = hf_ds.map(encode, remove_columns=hf_ds["train"].column_names)

# ---------- 5. Model + Trainer
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",     # run eval each epoch
    save_strategy="epoch",           # save checkpoints each epoch
    logging_strategy="steps",        # log every N steps
    logging_steps=50,                # adjust as needed
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_ds['train'],
    eval_dataset=encoded_ds['test'],
    tokenizer=tokenizer
)

# ---------- 6. Train
trainer.train()

# ---------- 7. Save
save_path = r"C:/Users/explo/Desktop/Mental Health Chatbot/backend/mental_health_chatbot"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ Multi-turn model saved to {save_path}")
