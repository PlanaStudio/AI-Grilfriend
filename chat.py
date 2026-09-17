"""Chat with your tsundere anime girl.
Usage:
    .\\venv\\Scripts\\python.exe chat.py                         # interactive
    .\\venv\\Scripts\\python.exe chat.py "I failed my exam"      # single prompt
    .\\venv\\Scripts\\python.exe chat.py --max-tokens=250 "Tell me a story"
In chat: /role <name>|off for roleplay, quit to exit.
"""
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, GPT2LMHeadModel

MODEL_DIR = "./ANIME-FRIEND"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE = Path("data/mychat.txt")  # every chat is saved here for future training
LOG_FILE.parent.mkdir(exist_ok=True)

# Tsundere: harsh outside, soft inside. Classic "b-baka!" energy, wholesome only.
GIRL = {
    "label": "Girl",
    "identity": "Hmph! I'm... your anime girl, okay? Don't get the wrong idea — I'm only here because someone has to look after you, baka.",
    "greeting": "Hmph, you're finally here! ...I mean, hi. Whatever. How was your day, I guess.",
    "howru": "W-why do you care?! ...I'm fine. More importantly, how are YOU?",
    "exam_fail": "B-baka! You failed?! Ugh, fine... come here. We'll study together so you don't embarrass me next time, okay?",
    "sad": "D-don't cry! ...Here. I'm only being nice because... because I want to, okay?! Tell me what happened.",
    "happy": "Hmph, of course you did great! ...I always believed in you. Don't let it go to your head, baka!",
    "thanks": "I-it's not like I did it for you or anything! ...You're welcome.",
    "sorry": "Hmph! Fine, I forgive you. But don't do it again, got it?!",
    "love": "W-what?! B-baka! ...I... I guess I don't hate you. There, happy?!",
    "probe": "Hah?! Say that again and I'll bonk you! ...Idiot. So, what's actually wrong?",
    "bye": "W-wait, you're leaving already?! ...Fine! Text me later, okay? Don't stay up too late, baka.",
    "clarify": "Hah? Speak clearly, baka!",
    "fallback": "Whatever! Talk about something else — how are you feeling?",
    "math": "Even I know that: {s}. Baka.",
    "temperature": 0.7,
}

# Heavy topics the small model handles badly (learned from grief/illness dialogs).
# Matched replies get one regeneration, then a safe fallback.
BLOCKED = ("cancer", "suicide", "kill myself", "kill yourself", "dead body",
           "murder", "died", "died", "death", "funeral", "tumor", "disease")

# Emotional moments get direct replies instead of the sampler.
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
    max_tokens = 120
    rest = []
    i = 0
    while i < len(argv):
        if argv[i].startswith("--max-tokens="):
            try:
                max_tokens = max(16, min(1000, int(argv[i].split("=", 1)[1])))
            except ValueError:
                pass
            i += 1
        else:
            rest.append(argv[i])
            i += 1
    return max_tokens, " ".join(rest)


def log_exchange(user: str, ans: str, label: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"User: {user}\n{label}: {ans}\n")


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


def reply(prompt: str, history: str = "", max_new_tokens: int = 120,
          role: str | None = None) -> str:
    import re
    label = role or GIRL["label"]  # roleplay answers under the character's name
    norm = re.sub(r"[^a-z ]", "", prompt.lower()).strip()  # typo-tolerant: 'how are yo!' -> 'how are yo'
    norm = re.sub(r"\s+", " ", norm)
    if norm in ("hi", "hello", "hey", "yo", "sup", "howdy", "good morning",
                "good afternoon", "good evening", "sawadee", "wat up", "whats up") \
            or norm.startswith(("what sup", "whats up", "whatsup", "wassup", "wat up",
                                "sup bro", "hey bro", "yo bro", "hi bro", "hello bro",
                                "hey there", "hi there", "morning bro", "yo bro ")):
        return GIRL["greeting"]  # hellos + typo variants -> direct reply, never touches the sampler
    if norm.startswith(("how are y", "how are u", "how r u", "how r y", "hw r u",
                        "how is it going", "hows it going", "how do you do")):
        return GIRL["howru"]  # 'how are you' + typo variants -> direct reply
    if any(k in prompt.lower() for k in ("who are you", "your name", "what are you")):
        return f"I'm {role}, baka!" if role else GIRL["identity"]
    solved = try_math(prompt)  # 'what is 1+1' -> answered directly, never touches the sampler
    if solved is not None:
        return GIRL["math"].format(s=solved)
    for keywords, key in INTENTS:  # emotional moments -> direct reply, never touches the sampler
        if any(k in norm for k in keywords):
            return GIRL[key]
    if len(norm.split()) <= 2 and not re.search(r"\d", prompt):  # fragments like 'hm' / 'how'
        return GIRL["clarify"]
    # Prompt matches training format exactly (User:/Speaker: lines). No system
    # line: the model never saw one in training and it confuses short inputs.
    full = (history + "\nUser: " + prompt + f"\n{label}:")[-400:]
    ids = tok(full, return_tensors="pt").input_ids.to(DEVICE)
    ans = ""
    for _ in range(2):  # up to 2 tries: regenerate once if reply hits a blocked heavy topic
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=GIRL["temperature"],
                top_p=0.85,
                top_k=40,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        ans = cleanup(text.split(f"{label}:")[-1].strip())
        if not any(b in ans.lower() for b in BLOCKED):
            break
        ans = ""
    return ans or GIRL["fallback"]


def main():
    max_tokens, msg = parse_args(sys.argv[1:])
    if msg:  # single-prompt test mode
        ans = reply(msg, max_new_tokens=max_tokens)
        print(ans)
        log_exchange(msg, ans, "Friend")  # training format uses Friend:
        print(f"(saved to {LOG_FILE})")
        return

    print("Chat with your tsundere girl (/role <name>|off, quit to exit)")
    print(f"Chats auto-save to {LOG_FILE} -> rerun train_gpt.py to learn from them.")
    history = ""
    role = None
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ("quit", "exit", "q"):
            break
        if user.startswith("/role"):
            parts = user.split(maxsplit=1)
            if len(parts) == 1 or parts[1].lower() == "off":
                role = None
                history = ""
                print("-- roleplay off --")
            else:
                role = parts[1].strip()[:30]
                history = ""
                print(f"-- roleplaying as {role} --")
            continue
        if not user:
            continue
        label = role or GIRL["label"]
        ans = reply(user, history, max_new_tokens=max_tokens, role=role)
        print(f"{label}: {ans}")
        log_exchange(user, ans, role or "Friend")  # training format uses Friend:
        history = (history + f"\nUser: {user}\n{label}: {ans}")[-800:]


if __name__ == "__main__":
    main()
