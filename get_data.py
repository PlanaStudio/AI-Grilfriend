"""Download conversation datasets from Hugging Face Hub for ANIME-FRIEND training.
Usage:
    .\\venv\\Scripts\\python.exe get_data.py list                        # show recommended chat datasets
    .\\venv\\Scripts\\python.exe get_data.py search <keyword>            # search Hub, e.g. 'dailydialog'
    .\\venv\\Scripts\\python.exe get_data.py get <dataset_id> [--split train] [--max 3000]
        # downloads and converts to data/chat_<name>.txt in "User: ... / Friend: ..." lines
"""
import sys
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RECOMMENDED = [
    ("pixelsandpointers/better_daily_dialog", "Everyday chit-chat, 87k rows. Best first pick for friend style. [verified]"),
    ("pixelsandpointers/empathetic_dialogues_for_lm", "Emotional conversations. Best pick for caring girlfriend mode. [verified]"),
    ("nadil-dulnidu/ai-girlfriend-chat-dataset", "480 girlfriend Q&A pairs, already downloaded to data/. [verified]"),
    ("RX5950XT/silicon-girlfriend-dataset", "Girlfriend conversations (conversations_clean.jsonl)."),
    ("Arsture/ideal-girlfriend", "Girlfriend-style data (parquet)."),
]

def cmd_list():
    print("Recommended chat datasets (use: get_data.py get <id>):\n")
    for ds_id, why in RECOMMENDED:
        print(f"  {ds_id}\n    {why}")

def cmd_search(keyword: str):
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Top matches for '{keyword}':")
    for d in list(api.list_datasets(search=keyword, limit=15)):
        print(f"  {d.id}  (downloads: {d.downloads})")

def dialog_to_lines(ds, max_n: int):
    """Auto-detect common dialog formats -> ['User: ...', 'Friend: ...', ...]."""
    lines: list[str] = []
    # Format A: one row = one utterance, grouped by dialog_id (e.g. better_daily_dialog)
    cols = set(ds.column_names or [])
    if {"dialog_id", "utterance"} <= cols:
        turn = 0
        for ex in ds:
            t = str(ex["utterance"]).strip().replace("\n", " ")
            if not t:
                continue
            who = "User" if turn % 2 == 0 else "Friend"
            lines.append(f"{who}: {t}")
            turn += 1
            if len(lines) >= max_n:
                break
        return lines
    count = 0
    for ex in ds:
        if count >= max_n:
            break
        turns = None
        if {"question", "answer"} <= set(ex.keys()):  # Q&A pairs (e.g. ai-girlfriend-chat-dataset)
            turns = [f"User: {str(ex['question']).strip()}", f"Friend: {str(ex['answer']).strip()}"]
            lines.extend(t for t in turns if t.split(': ', 1)[1])
            count += 1
            continue
        if "conv" in ex and isinstance(ex["conv"], list):  # empathetic_dialogues_for_lm
            turns = [str(t).replace("_comma_", ",") for t in ex["conv"]]
        elif "dialog" in ex and isinstance(ex["dialog"], list):  # daily_dialog
            turns = ex["dialog"]
        elif "utterances" in ex and isinstance(ex["utterances"], list):  # empathetic_dialogues
            u = ex["utterances"]
            turns = [t["utterance"] if isinstance(t, dict) else t for t in u]
        elif "history" in ex and isinstance(ex["history"], list):  # persona_chat
            turns = list(ex["history"]) + [ex.get("candidates", [""])[-1] if ex.get("candidates") else ""]
        elif "text" in ex and isinstance(ex["text"], str):
            turns = [ex["text"]]
        if not turns:
            continue
        for i, t in enumerate(turns):
            t = str(t).strip().replace("\n", " ")
            if not t:
                continue
            who = "User" if i % 2 == 0 else "Friend"
            lines.append(f"{who}: {t}")
            count += 1
            if count >= max_n:
                break
    return lines

def cmd_get(ds_id: str, split: str = "train", max_n: int = 3000):
    from huggingface_hub import HfApi
    from datasets import load_dataset

    print(f"Downloading {ds_id} (split={split})...")

    repo_files = HfApi().list_repo_files(ds_id, repo_type="dataset")
    data_extensions = (".json", ".jsonl", ".csv", ".tsv", ".parquet", ".arrow", ".txt")
    data_files = [path for path in repo_files if path.lower().endswith(data_extensions)]
    if not data_files:
        raise RuntimeError(
            f"{ds_id} does not contain conversation data files. "
            "It contains model/assets files (for example models.zip), not a text dataset. "
            "Use get_data.py list or provide a dataset with JSON, CSV, Parquet, or TXT files."
        )

    ds = load_dataset(ds_id, split=split, trust_remote_code=False)
    print(f"Rows: {len(ds)}, columns: {ds.column_names}")
    lines = dialog_to_lines(ds, max_n)
    out = DATA_DIR / f"chat_{ds_id.replace('/', '_')}.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} lines -> {out}")

def main():
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        cmd_list()
    elif sys.argv[1] == "search" and len(sys.argv) > 2:
        cmd_search(sys.argv[2])
    elif sys.argv[1] == "get" and len(sys.argv) > 2:
        split, max_n = "train", 3000
        for a in sys.argv[3:]:
            if a.startswith("--split="):
                split = a.split("=", 1)[1]
            elif a.startswith("--max="):
                max_n = int(a.split("=", 1)[1])
        cmd_get(sys.argv[2], split, max_n)
    else:
        print(__doc__)

if __name__ == "__main__":
    main()
