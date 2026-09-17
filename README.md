# Ex-Friend

A small GPT-style friend / girlfriend chatbot you train and run locally on your own GPU.
Decoder-only Transformer (GPT-2 family, same idea as GPT-3), fine-tuned on your chats
plus public dialog corpora. Tuned for a 4GB VRAM laptop GPU (RTX 3050).

## Setup

```powershell
# 1. Virtual env (Python 3.10)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. PyTorch with CUDA 12.1 (CPU works but is ~10-50x slower)
.\venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Rest of the stack
.\venv\Scripts\pip.exe install -r requirements.txt
```

## Get training data

```powershell
.\venv\Scripts\python.exe get_data.py list                 # recommended chat datasets
.\venv\Scripts\python.exe get_data.py search <keyword>     # search Hugging Face Hub
.\venv\Scripts\python.exe get_data.py get <dataset_id> --max=3000   # -> data/chat_<name>.txt
```

Verified picks: `pixelsandpointers/better_daily_dialog` (friend chit-chat),
`pixelsandpointers/empathetic_dialogues_for_lm` (caring tone),
`nadil-dulnidu/ai-girlfriend-chat-dataset` (girlfriend Q&A).
Your own exports go through `clean_thai.py` (LINE chat -> `english_only.txt`).

## Train

Edit `DATA_FILES` / `EPOCHS` at the top of `train_gpt.py`, then:

```powershell
.\venv\Scripts\python.exe train_gpt.py
```

- `FROM_SCRATCH = False` (default): fine-tune `distilgpt2` — keeps English grammar.
  `True` builds a 30M model from random weights (needs lots of data).
- Junk link/ID lines (`http`, `tiktok`, `.com`) are filtered automatically.
- Saves to `./Ex-friend/` (weights excluded from git — too large for GitHub, retrain to rebuild).

## Chat

```powershell
.\venv\Scripts\python.exe chat.py                                        # friend mode
.\venv\Scripts\python.exe chat.py --mode gf                              # girlfriend mode
.\venv\Scripts\python.exe chat.py --mode friend "I failed my exam"       # single prompt
.\venv\Scripts\python.exe chat.py --mode friend --max-tokens=250 "Tell me a story"
```

In-chat commands: `/mode friend` | `/mode gf` to switch personality, `quit` to exit.

Small-talk (`hi`, `how are you` + typos), identity questions, and emotional
moments (failed exam, bad news, thanks, bye...) get direct replies; heavier
topics are filtered with regenerate-then-fallback. Open-ended replies come
from the model with repetition + garbage-token cleanup.

## Files

| File | Purpose |
|---|---|
| `chat.py` | Friend/girlfriend chat + guardrails |
| `train_gpt.py` | Fine-tune causal LM on all data |
| `train.py` | Classifier starter (IMDb sentiment) |
| `get_data.py` | Download + convert Hub dialog datasets |
| `clean_thai.py` | Split LINE export into Thai/English text |
| `requirements.txt` | Pinned install list |

## Notes

- 4GB VRAM: keep batch ≤ 8, context ≤ 256, prefer small models.
- Thai data (`thai_only.txt`) is excluded from training — the English tokenizer mangles it.
