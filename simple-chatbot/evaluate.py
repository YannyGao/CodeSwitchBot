import json
from tqdm import tqdm
import os
from chatbot import LocalChat  # or TogetherChat

# -------------------------
# Config
# -------------------------
DATA_PATH = "simple-chatbot/sample_sentences.json"
OUT_PATH = "qwen_pomdp_outputs.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
USE_POMDP = True
CHUNK_SIZE = 20  # messages per mini-conversation

# -------------------------
# Prepare output file
# -------------------------
with open(OUT_PATH, "w", encoding="utf-8") as f:
    pass

# -------------------------
# Load dataset
# -------------------------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    sentences = json.load(f)

print(f"Loaded {len(sentences)} sentences.")

# -------------------------
# Split dataset into chunks
# -------------------------
chunks = [sentences[i:i+CHUNK_SIZE] for i in range(0, len(sentences), CHUNK_SIZE)]

# -------------------------
# Initialize chatbot (once!)
# -------------------------
print("Initializing chatbot...")
bot = LocalChat(model=MODEL_NAME, use_pomdp=USE_POMDP)
device = "cuda" if hasattr(bot.pipe.model, "device") and str(bot.pipe.model.device) != "cpu" else "cpu"
print(f"Using device: {device}")

results = []

# -------------------------
# Run chatbot
# -------------------------
for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
    # Reset POMDP for new mini-conversation
    if USE_POMDP:
        bot.reset()

    for text in tqdm(chunk, desc=f"Chunk {chunk_idx+1}", leave=False):
        try:
            reply = bot.chat(text)
        except Exception as e:
            reply = f"[ERROR] {e}"

        row = {
            "input_text": text,
            "output_text": reply,
            "pomdp_action": bot.get_last_action(),
            "user_cmi": bot.get_last_cmi(),
            "belief": bot.get_belief(),
        }

        results.append(row)

        # Incremental save
        with open(OUT_PATH, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

print(f"\n✅ Finished! Results saved to {OUT_PATH}")
