import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# -------------------------
# Load Together outputs
# -------------------------
FILE_PATH = "mistral_pomdp_outputs.jsonl"

rows = []
with open(FILE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

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
    zh = sum(1 for _, l in tagged if l == "zh")
    en = sum(1 for _, l in tagged if l == "en")
    total = zh + en
    if total == 0:
        return 0
    return min(zh, en) / total

# -------------------------
# Build DataFrame with turn-level CMI
# -------------------------
turn_data = []
for row in rows:
    conv_idx = row["conversation_idx"]
    turn_idx = row["turn_idx"]
    input_text = row["input_text"]
    output_text = row["output_text"]

    input_cmi = row.get("input_cmi", None)
    if input_cmi is None:
        input_cmi = cmi_from_tags(tag_sentence(input_text)) * 100

    output_cmi = cmi_from_tags(tag_sentence(output_text)) * 100

    turn_data.append({
        "conversation_idx": conv_idx,
        "turn_idx": turn_idx,
        "input_cmi": input_cmi,
        "output_cmi": output_cmi
    })

df_turns = pd.DataFrame(turn_data)

# -------------------------
# Plot convergence over time for each conversation
# -------------------------
plt.figure(figsize=(10,6))

for conv_idx, group in df_turns.groupby("conversation_idx"):
    plt.plot(group["turn_idx"], group["input_cmi"], color="blue", alpha=0.3)
    plt.plot(group["turn_idx"], group["output_cmi"], color="red", alpha=0.3)

plt.xlabel("Turn index")
plt.ylabel("CMI (%)")
plt.title("CMI convergence over conversation turns (blue=user, red=chatbot)")
plt.grid(True)
plt.show()

# -------------------------
# Optionally: aggregate across conversations
# -------------------------
df_summary = df_turns.groupby("turn_idx").agg(
    mean_input_cmi = ("input_cmi", "mean"),
    mean_output_cmi = ("output_cmi", "mean")
).reset_index()

plt.figure(figsize=(10,6))
plt.plot(df_summary["turn_idx"], df_summary["mean_input_cmi"], label="Mean Input CMI", color="blue")
plt.plot(df_summary["turn_idx"], df_summary["mean_output_cmi"], label="Mean Output CMI", color="red")
plt.xlabel("Turn index")
plt.ylabel("CMI (%)")
plt.title("Average CMI over conversation turns")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Correlation per turn index (optional)
# -------------------------
pearson_r, _ = pearsonr(df_turns["input_cmi"], df_turns["output_cmi"])
spearman_rho, _ = spearmanr(df_turns["input_cmi"], df_turns["output_cmi"])
print(f"Overall Pearson r = {pearson_r:.3f}")
print(f"Overall Spearman ρ = {spearman_rho:.3f}")
