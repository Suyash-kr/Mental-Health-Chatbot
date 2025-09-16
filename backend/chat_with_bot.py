from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os


model_path = r"C:/Users/explo/Desktop/Mental Health Chatbot/backend/mental_health_chatbot"
# Load tokenizer and model
# load
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
model.eval()

# ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model.resize_token_embeddings(len(tokenizer))

MAX_INPUT_LEN = 128    # how many tokens of context we keep
MAX_NEW_TOKENS = 40    # how many tokens the model may generate (short replies)

def clean_artifacts(text: str) -> str:
    # Remove stray list markers, weird quotes, duplicated markers
    text = re.sub(r"\[\s*'|'\s*\]|\"\s*\"|\[\s*\"|\"\s*\]", " ", text)
    text = text.replace("']","").replace("['","").replace('"]','').replace('["',"")
    text = re.sub(r"<bot>[:]*", "<bot>:", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Remove odd leftover punctuation clusters at start/end
    text = re.sub(r'^[^A-Za-z0-9]+', '', text)
    text = re.sub(r'[^A-Za-z0-9\.\!\?]+$', '', text)
    return text.strip()

def first_n_sentences(text: str, n=2) -> str:
    # simple sentence splitter by punctuation — keeps at most n sentences
    parts = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    if not parts:
        return text.strip()
    keep = parts[:n]
    out = " ".join(s.rstrip() for s in keep).strip()
    # ensure ending punctuation
    if out and out[-1] not in ".!?":
        out += "."
    return out

print("Chatbot ready (type 'quit' to exit)\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit"):
        print("Bot: Take care — you can come back anytime.")
        break

    # Build prompt (single-turn; you can keep a short history if desired)
    prompt = f"<user>: {user_input} <bot>:"

    # Tokenize with attention mask (prevents the attention warning)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
        padding=True
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate -- conservative sampling + repetition prevention
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=0.85,
            temperature=0.6,
            top_k=50,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    # decode
    out_text = tokenizer.decode(generated[0], skip_special_tokens=False)
    # Keep only part after last <bot>:
    if "<bot>:" in out_text:
        out_text = out_text.split("<bot>:")[-1]
    # remove any accidental user marker trailing
    out_text = out_text.split("<user>:")[0]

    out_text = clean_artifacts(out_text)
    # Keep only first 1-2 sentences for brevity & safety
    out_text = first_n_sentences(out_text, n=2)

    print("Bot:", out_text)