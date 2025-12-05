import json
from tqdm import tqdm
import re
from chatbot import MistralChat  # import your class

# -------------------------
# CMI helpers
# -------------------------
def tag_sentence(text):
    zh = re.compile(r"[\u4e00-\u9fff]")
    en = re.compile(r"[a-zA-Z]")
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+|[^\s]", text)
    out = []
    for t in tokens:
        if zh.search(t):
            out.append((t, "zh"))
        elif en.search(t):
            out.append((t, "en"))
        else:
            out.append((t, "other"))
    return out

def cmi_from_tags(tagged):
    """
    Code-Mixing Index in PERCENT (0–50).

    CMI = 100 * min(en, zh) / (en + zh)
    """
    zh = sum(1 for _, l in tagged if l == "zh")
    en = sum(1 for _, l in tagged if l == "en")
    total = zh + en
    if total == 0:
        return 0.0
    return 100.0 * min(zh, en) / total

# -------------------------
# Load utterances
# -------------------------
DATA_PATH = "simple-chatbot/utterances_by_speaker_session.json"
OUT_PATH = "mistral_pomdp_outputs.jsonl"
CHUNK_SIZE = 20

with open(DATA_PATH, "r", encoding="utf-8") as f:
    speaker_sessions = json.load(f)

# Flatten and chunk: each chunk = one "conversation" (up to 20 utterances)
conversations = []
for key, utterances in speaker_sessions.items():
    for i in range(0, len(utterances), CHUNK_SIZE):
        chunk = utterances[i:i + CHUNK_SIZE]
        chunk_cmi = [
            {"text": s, "cmi": cmi_from_tags(tag_sentence(s))}
            for s in chunk
        ]
        conversations.append(chunk_cmi)

print(f"Prepared {len(conversations)} conversations.")

# -------------------------
# Load existing results (if any)
# -------------------------
processed_turns = set()
try:
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            processed_turns.add((row["conversation_idx"], row["turn_idx"]))
except FileNotFoundError:
    pass

print(f"Found {len(processed_turns)} already processed turns. Will skip them.")

# -------------------------
# Initialize Mistral chatbot
# -------------------------
bot = MistralChat(model="mistral-medium-latest", use_pomdp=True)

# -------------------------
# Run conversations
# -------------------------
for conv_idx, conversation in enumerate(tqdm(conversations, desc="Processing conversations")):
    bot.reset()
    for turn_idx, msg in enumerate(conversation):
        if (conv_idx, turn_idx) in processed_turns:
            continue  # skip already processed

        text = msg["text"]
        input_cmi = msg["cmi"]  # already in percent

        try:
            reply = bot.chat(text)
        except Exception as e:
            reply = f"[ERROR] {e}"

        output_cmi = cmi_from_tags(tag_sentence(reply))  # percent

        row = {
            "conversation_idx": conv_idx,
            "turn_idx": turn_idx,
            "input_text": text,
            "output_text": reply,
            "input_cmi": input_cmi,
            "output_cmi": output_cmi,
            "pomdp_action": bot.get_last_action(),
            "belief": bot.get_belief(),
        }

        # Incremental save
        with open(OUT_PATH, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

print(f"✅ Finished! Results saved to {OUT_PATH}")
