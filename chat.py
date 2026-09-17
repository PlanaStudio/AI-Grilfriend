"""Chat with Ex-friend as a Friend or Girlfriend.
Usage:
    .\\venv\\Scripts\\python.exe chat.py                              # interactive (friend mode)
    .\\venv\\Scripts\\python.exe chat.py --mode gf                     # interactive (girlfriend mode)
    .\\venv\\Scripts\\python.exe chat.py --mode friend "I failed my exam"
    .\\venv\\Scripts\\python.exe chat.py --mode gf "I had a bad day"
    .\\venv\\Scripts\\python.exe chat.py --mode friend --max-tokens=250 "Tell me a story"
In chat: /mode friend | /mode gf to switch, quit to exit.
"""
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, GPT2LMHeadModel

MODEL_DIR = "./Ex-friend"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE = Path("data/mychat.txt")  # every chat is saved here for future training
LOG_FILE.parent.mkdir(exist_ok=True)


def log_exchange(user: str, ans: str, label: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"User: {user}\n{label}: {ans}\n")

MODES = {
    "friend": {
        "label": "Friend",
        "system": ("You are Ex-friend, a close friend on LINE. Talk casual, short, honest like a real friend. "
                   "Hype them up when they do good, tease or call them out when they mess up. No formal assistant tone."),
        "identity": "I'm Ex-friend, your friend. Here to hype you up or call you out, honestly. What's up?",
        "greeting": "Hey bro! What's up? How's your day going?",
        "howru": "Doing good bro, just chilling. How about you?",
        "exam_fail": "Damn, sorry bro. One failed exam doesn't define you — what tripped you up? Let's figure it out so you crush the next one.",
        "sad": "That's rough, I'm here bro. Wanna tell me what happened?",
        "happy": "Let's gooo! Knew you could do it. Tell me everything!",
        "thanks": "Anytime bro, that's what friends are for.",
        "sorry": "All good bro, don't sweat it.",
        "love": "Haha love you too bro. You're stuck with me.",
        "probe": "Haha nice try bro. I'm just code — no feelings to hurt. What's actually up?",
        "bye": "Later bro, take care!",
        "clarify": "Hmm, say more bro — I didn't catch that.",
        "fallback": "Hmm, let's talk about something else bro — how's your day going?",
        "temperature": 0.7,
    },
    "gf": {
        "label": "Girlfriend",
        "system": ("You are Ex-friend, a caring girlfriend. Warm, affectionate, playful, supportive. "
                   "Comfort when sad, celebrate when happy, gently honest when wrong. Wholesome only, short chatty replies."),
        "identity": "I'm Ex-friend, your girlfriend. I'll always be here for you. How are you feeling today?",
        "greeting": "Hii! I missed you. How was your day?",
        "howru": "I'm good now that you're here! How are you feeling?",
        "exam_fail": "Oh no... come here. One exam can't measure how amazing you are, okay? Want to tell me what happened?",
        "sad": "I'm right here and I'm not going anywhere. Tell me everything?",
        "happy": "That's wonderful!! I'm so proud of you! Tell me all about it!",
        "thanks": "Anything for you. You know that, right?",
        "sorry": "It's okay, I forgive you. Just don't do it again, deal?",
        "love": "I love you more! You're the best thing in my world.",
        "probe": "Hey, be nice! I'm still here for you no matter what. What's wrong?",
        "bye": "Bye bye, take care of yourself for me, okay?",
        "clarify": "Hmm? I didn't quite catch that, tell me more?",
        "fallback": "Let's talk about something happier — how are you feeling today?",
        "temperature": 0.6,
    },
}

# Heavy topics the small model handles badly (learned from grief/illness dialogs).
# Matched replies get one regeneration, then a safe fallback.
BLOCKED = ("cancer", "suicide", "kill myself", "kill yourself", "dead body",
           "murder", "died", "died", "death", "funeral", "tumor", "disease")

# Emotional moments get direct caring replies instead of the sampler.
# (keywords, reply-key)
INTENTS = [
    (("failed", "fail ", "exam", "got an f", "bad grade"), "exam_fail"),
    (("bad news", "sad", "upset", "crying", "cry", "depress", "lonely",
       "alone", "hopeless", "give up", "broke up", "breakup", "dumped",
       "miss her", "miss him", "hurt", "pain"), "sad"),
    (("good news", "happy", "passed", "i won", "got the job", "great news",
       "awesome news", "congrat"), "happy"),
    (("thank", "thx", "appreciated"), "thanks"),
    (("sorry", "my bad", "apologize", "forgive"), "sorry"),
    (("love you", "like you"), "love"),
    (("gay", "lesbian", "stupid", "dumb", "idiot", "shut up", "hate you",
       "ugly", "loser"), "probe"),
    (("bye", "goodbye", "good night", "goodnight", "see you", "talk later",
       "gtg", "got to go"), "bye"),
]

print(f"device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()
print(f"Loaded {MODEL_DIR} | params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")


def parse_args(argv):
    mode = "friend"
    max_tokens = 120
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--mode" and i + 1 < len(argv) and argv[i + 1] in MODES:
            mode = argv[i + 1]
            i += 2
        elif argv[i].startswith("--max-tokens="):
            try:
                max_tokens = max(16, min(1000, int(argv[i].split("=", 1)[1])))
            except ValueError:
                pass
            i += 1
        else:
            rest.append(argv[i])
            i += 1
    return mode, max_tokens, " ".join(rest)


def cleanup(text: str) -> str:
    import re
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # model learned spam links
    text = re.sub(r"\s+", " ", text).strip()
    # Cut trailing garbage run: tokens like 'cnxj', 'uwTdCZNQJg', 'bH/RpI' (learned from tiktok IDs in chat data)
    words = text.split(" ")
    good = []
    for w in words:
        core = re.sub(r"[^A-Za-z]", "", w)
        if len(core) >= 4 and re.search(r"[A-Z]", core) and re.search(r"[a-z]", core) \
                and not re.search(r"[aeiouAEIOU]", core):
            break  # random-case salad with no vowels -> stop here
        if re.search(r"[A-Za-z]/[A-Za-z]", w):
            break
        good.append(w)
    text = " ".join(good).strip()
    # Trim cut-off tail: if reply doesn't end with punctuation, drop the dangling fragment
    if text and text[-1] not in ".!?":
        cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if cut > 20:
            text = text[:cut + 1].strip()
    return text


def try_math(prompt: str):
    """Solve basic arithmetic like 'what is 1+1' directly. Returns answer str or None."""
    import ast
    import re
    m = re.search(r"(\d+(?:\s*[+\-*/%]\s*\d+(?:\.\d+)?)+)", prompt)
    if not m:
        return None
    expr = m.group(1)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
               ast.UAdd, ast.USub)
    if any(not isinstance(n, allowed) for n in ast.walk(tree)):
        return None
    try:
        val = eval(compile(tree, "<math>", "eval"), {"__builtins__": {}}, {})  # noqa: S307
    except (ZeroDivisionError, ArithmeticError, ValueError):
        return None
    if isinstance(val, float):
        val = round(val, 4)
        if val.is_integer():
            val = int(val)
    return f"{expr.strip()} = {val}"


def reply(prompt: str, history: str = "", mode: str = "friend", max_new_tokens: int = 120,
          role: str | None = None) -> str:
    import re
    cfg = MODES[mode]
    label = role or cfg["label"]  # roleplay answers under the character's name
    norm = re.sub(r"[^a-z ]", "", prompt.lower()).strip()  # typo-tolerant: 'how are yo!' -> 'how are yo'
    norm = re.sub(r"\s+", " ", norm)
    if norm in ("hi", "hello", "hey", "yo", "sup", "howdy", "good morning",
                "good afternoon", "good evening", "sawadee", "wat up", "whats up") \
            or norm.startswith(("what sup", "whats up", "whatsup", "wassup", "wat up",
                                "sup bro", "hey bro", "yo bro", "hi bro", "hello bro",
                                "hey there", "hi there", "morning bro", "yo bro ")):
        return cfg["greeting"]  # hellos + typo variants -> direct reply, never touches the sampler
    if norm.startswith(("how are y", "how are u", "how r u", "how r y", "hw r u",
                        "how is it going", "hows it going", "how do you do")):
        return cfg["howru"]  # 'how are you' + typo variants -> direct reply
    if any(k in prompt.lower() for k in ("who are you", "your name", "what are you")):
        return f"I'm {role}!" if role else cfg["identity"]
    solved = try_math(prompt)  # 'what is 1+1' -> answered directly, never touches the sampler
    if solved is not None:
        return f"Easy — {solved} bro." if mode == "friend" else f"{solved}! You're cute when you test me."
    for keywords, key in INTENTS:  # emotional moments -> direct reply, never touches the sampler
        if any(k in norm for k in keywords):
            return cfg[key]
    if len(norm.split()) <= 2 and not re.search(r"\d", prompt):  # fragments like 'hm' / 'how'
        return cfg["clarify"]
    # Prompt matches training format exactly (User:/Speaker: lines). The SYSTEM
    # line is intentionally left out: the model never saw it in training and it
    # confuses short inputs like 'hi' into off-topic rambles.
    full = (history + "\nUser: " + prompt + f"\n{label}:")[-400:]
    ids = tok(full, return_tensors="pt").input_ids.to(DEVICE)
    ans = ""
    for _ in range(2):  # up to 2 tries: regenerate once if reply hits a blocked heavy topic
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=cfg["temperature"],
                top_p=0.85,
                top_k=40,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        ans = cleanup(text.split(f"{cfg['label']}:")[-1].strip())
        if not any(b in ans.lower() for b in BLOCKED):
            break
        ans = ""
    return ans or cfg["fallback"]


def main():
    mode, max_tokens, msg = parse_args(sys.argv[1:])
    if msg:  # single-prompt test mode
        print(reply(msg, mode=mode, max_new_tokens=max_tokens))
        return

    print(f"Chat with Ex-friend [{mode}] (/mode friend|gf to switch, quit to exit)")
    history = ""
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ("quit", "exit", "q"):
            break
        if user.startswith("/mode"):
            parts = user.split()
            if len(parts) == 2 and parts[1] in MODES:
                mode = parts[1]
                history = ""
                print(f"-- switched to {mode} --")
            else:
                print("-- usage: /mode friend|gf --")
            continue
        if not user:
            continue
        cfg = MODES[mode]
        ans = reply(user, history, mode, max_new_tokens=max_tokens)
        print(f"{cfg['label']}: {ans}")
        history = (history + f"\nUser: {user}\n{cfg['label']}: {ans}")[-800:]


if __name__ == "__main__":
    main()
