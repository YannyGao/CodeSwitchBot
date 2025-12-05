import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# -------------------------
# Load outputs
# -------------------------
FILE_PATH = "mistral_pomdp_outputs.jsonl"

rows = []
with open(FILE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

# -------------------------
# CMI helpers (for safety)
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
        return 0.0
    return 100.0 * min(zh, en) / total  # percent

# -------------------------
# Build DataFrame with turn-level CMI
# -------------------------
turn_data = []
for row in rows:
    conv_idx = row["conversation_idx"]
    turn_idx = row["turn_idx"]
    input_text = row["input_text"]
    output_text = row["output_text"]

    # Use stored CMI if present, otherwise recompute
    input_cmi = row.get("input_cmi")
    if input_cmi is None:
        input_cmi = cmi_from_tags(tag_sentence(input_text))

    output_cmi = row.get("output_cmi")
    if output_cmi is None:
        output_cmi = cmi_from_tags(tag_sentence(output_text))

    turn_data.append({
        "conversation_idx": conv_idx,
        "turn_idx": turn_idx,
        "input_cmi": input_cmi,
        "output_cmi": output_cmi,
    })

df_turns = pd.DataFrame(turn_data)
df_turns = df_turns.sort_values(["conversation_idx", "turn_idx"]).reset_index(drop=True)

# ----------------------------------------------------
# 1) Per-turn visualization (what you plotted before)
# ----------------------------------------------------
plt.figure(figsize=(10, 6))

for conv_idx, group in df_turns.groupby("conversation_idx"):
    group = group.sort_values("turn_idx")
    plt.plot(group["turn_idx"], group["input_cmi"], color="blue", alpha=0.3)
    plt.plot(group["turn_idx"], group["output_cmi"], color="red", alpha=0.3)

plt.xlabel("Turn index")
plt.ylabel("CMI (%)")
plt.title("CMI convergence over conversation turns (blue=user, red=chatbot)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# 2) Average CMI over turns (still per-turn index)
# ----------------------------------------------------
df_summary = df_turns.groupby("turn_idx").agg(
    mean_input_cmi=("input_cmi", "mean"),
    mean_output_cmi=("output_cmi", "mean"),
).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(df_summary["turn_idx"], df_summary["mean_input_cmi"],
         label="Mean Input CMI", color="blue")
plt.plot(df_summary["turn_idx"], df_summary["mean_output_cmi"],
         label="Mean Output CMI", color="red")
plt.xlabel("Turn index")
plt.ylabel("CMI (%)")
plt.title("Average CMI over conversation turns")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# 3) Conversation-level correlation (what you want)
#    avg input CMI vs average of last 3 output turns
# ----------------------------------------------------
conv_stats = []
for conv_idx, group in df_turns.groupby("conversation_idx"):
    group = group.sort_values("turn_idx")
    avg_input = group["input_cmi"].mean()
    last_k = min(3, len(group))
    final_output = group.tail(last_k)["output_cmi"].mean()

    conv_stats.append({
        "conversation_idx": conv_idx,
        "avg_input_cmi": avg_input,
        "final_output_cmi": final_output,
    })

df_conv = pd.DataFrame(conv_stats)

# Scatter: conversation-level
plt.figure(figsize=(6, 6))
plt.scatter(df_conv["avg_input_cmi"], df_conv["final_output_cmi"], alpha=0.6)
plt.xlabel("Average Input CMI (%)")
plt.ylabel("Final Output CMI (last 3 turns, %)")
plt.title("Conversation-level Input vs Output CMI")
plt.grid(True)
plt.tight_layout()
plt.show()

# Correlations
pearson_r, _ = pearsonr(df_conv["avg_input_cmi"], df_conv["final_output_cmi"])
spearman_rho, _ = spearmanr(df_conv["avg_input_cmi"], df_conv["final_output_cmi"])
print(f"Conversation-level Pearson r = {pearson_r:.3f}")
print(f"Conversation-level Spearman ρ = {spearman_rho:.3f}")

# (Optional) also print per-turn correlation if you still want to compare
pt_pearson, _ = pearsonr(df_turns["input_cmi"], df_turns["output_cmi"])
pt_spearman, _ = spearmanr(df_turns["input_cmi"], df_turns["output_cmi"])
print(f"Per-turn Pearson r = {pt_pearson:.3f}")
print(f"Per-turn Spearman ρ = {pt_spearman:.3f}")
