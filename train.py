"""Starter training script - optimized for RTX 3050 Laptop 4GB.
Usage:
    .\\venv\\Scripts\\python.exe train.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset


MODEL_NAME = "distilbert-base-uncased"  # small, fits 4GB VRAM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"torch {torch.__version__} | device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def main():
    # Example: IMDb sentiment - replace with your dataset
    ds = load_dataset("imdb")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(b):
        return tok(b["text"], truncation=True, padding="max_length", max_length=256)

    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=8,      # 4GB-safe; lower to 4 if OOM
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,      # effective batch 16
        num_train_epochs=2,
        fp16=True,                          # RTX 3050 supports FP16
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"].shuffle(seed=42).select(range(2000)),  # subset for quick test
        eval_dataset=ds["test"].select(range(500)),
        tokenizer=tok,
    )
    trainer.train()
    print("Done. Best model in ./checkpoints")


if __name__ == "__main__":
    main()
