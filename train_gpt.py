"""GPT-3-inspired causal LM training - decoder-only Transformer (GPT-2 family).
Same architecture idea as GPT-3 (masked self-attention, next-token prediction),
sized for RTX 3050 4GB.

Usage:
    .\\venv\\Scripts\\python.exe train_gpt.py

Requires CUDA PyTorch build (already installed: torch 2.5.1+cu121).
CPU works but is ~10-50x slower for Transformers.
"""
import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

BASE_TOKENIZER = "distilgpt2"  # tokenizer source (GPT-3-style BPE)
NEW_MODEL_DIR = "./Ex-friend"  # local output folder, NOT a HuggingFace repo id
DATA_FILES = [  # ALL training data: your LINE chat + downloaded dialog corpora
    "data/english_only.txt",
    "data/chat_pixelsandpointers_better_daily_dialog.txt",
    "data/chat_pixelsandpointers_empathetic_dialogues_for_lm.txt",
    "data/chat_nadil-dulnidu_ai-girlfriend-chat-dataset.txt",
]
EPOCHS = 3
FROM_SCRATCH = False  # False = fine-tune pretrained (keeps English grammar). True = random init (gibberish on tiny data).
JUNK = ("http", "tiktok", ".com", "www.")  # drop link/ID salad lines that teach gibberish

# GPU required for practical training speed. Falls back to CPU with warning.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} (cuda {torch.version.cuda}) | device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: running on CPU - expect extremely slow training. Install CUDA PyTorch for GPU.")


def main():
    raw = load_dataset("text", data_files=DATA_FILES, split="train")
    # Drop link/ID salad lines (e.g. 'mint https vt tiktok com ZSuWNPSM') that teach gibberish
    raw = raw.filter(lambda ex: not any(j in ex["text"].lower() for j in JUNK))
    print(f"Lines after junk filter: {len(raw)}")
    ds = raw.train_test_split(test_size=0.1, seed=42)
    tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    tok.pad_token = tok.eos_token  # GPT-2 has no pad token

    def tokenize(b):
        return tok(b["text"], truncation=True, max_length=128)

    tok_ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # False = fine-tune pretrained distilgpt2 (keeps grammar). True = random 30M init.
    # Never use from_pretrained("Ex-friend") here - that looks on HuggingFace Hub.
    if FROM_SCRATCH:
        config = GPT2Config(
            vocab_size=tok.vocab_size,
            n_positions=256,
            n_ctx=256,
            n_embd=384,
            n_layer=6,
            n_head=6,
        )
        model = GPT2LMHeadModel(config)
    else:
        model = GPT2LMHeadModel.from_pretrained(BASE_TOKENIZER)
    model.config.pad_token_id = tok.pad_token_id
    model.to(DEVICE)  # Trainer also moves it, explicit for clarity
    print(f"Model on: {next(model.parameters()).device} | params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    args = TrainingArguments(
        output_dir=NEW_MODEL_DIR,
        per_device_train_batch_size=4,   # 4GB-safe
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,   # effective batch 16
        num_train_epochs=EPOCHS,
        fp16=(DEVICE == "cuda"),          # Ampere RTX 3050: use fp16
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["test"],
        data_collator=collator,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(NEW_MODEL_DIR)
    tok.save_pretrained(NEW_MODEL_DIR)
    print(f"Done. Model saved in {NEW_MODEL_DIR}")

    # Quick generation smoke test
    model.eval()
    prompt = "Once upon a time"
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=30, do_sample=True, top_p=0.95, pad_token_id=tok.eos_token_id)
    print(tok.decode(out[0]))


if __name__ == "__main__":
    main()
