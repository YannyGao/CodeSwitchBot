import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, entropy

FILE_PATH = "mistral_pomdp_outputs.jsonl"

rows = []
with open(FILE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} rows from {FILE_PATH}")

# Sanity: required columns
required_cols = {"conversation_idx", "turn_idx", "input_cmi", "output_cmi", "pomdp_action", "belief"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in data: {missing}")


STATES = ["EN_DOMINANT", "MIXED", "ZH_DOMINANT"]

def expected_action_from_belief(belief_dict):
    """
    Map the highest-belief state to the 'ideal' action.
    EN_DOMINANT -> RESPOND_EN
    MIXED       -> MIXED_RESPONSE
    ZH_DOMINANT -> RESPOND_ZH
    """
    if not isinstance(belief_dict, dict) or len(belief_dict) == 0:
        return None
    # get state with max probability
    state = max(belief_dict, key=belief_dict.get)
    if state == "EN_DOMINANT":
        return "RESPOND_EN"
    elif state == "ZH_DOMINANT":
        return "RESPOND_ZH"
    else:
        return "MIXED_RESPONSE"

df["expected_action"] = df["belief"].apply(expected_action_from_belief)
mask_valid = df["expected_action"].notna()

belief_action_agreement = (df.loc[mask_valid, "expected_action"] == df.loc[mask_valid, "pomdp_action"]).mean()

print("\n=== Belief–Action Alignment ===")
print(f"Belief–action agreement: {belief_action_agreement:.3f} ({belief_action_agreement*100:.1f}%)")


def kl_per_conversation(group, n_bins=10, max_cmi=0.5):
    """
    Compute KL(P || Q) where:
      P = histogram of input_cmi
      Q = histogram of output_cmi
    Both smoothed and normalized.
    """
    inp = group["input_cmi"].values
    out = group["output_cmi"].values

    # define bins from 0 to max_cmi (inclusive)
    bins = np.linspace(0.0, max_cmi, n_bins + 1)

    p_counts, _ = np.histogram(inp, bins=bins)
    q_counts, _ = np.histogram(out, bins=bins)

    # add small smoothing to avoid zeroes
    p = p_counts.astype(float) + 1e-6
    q = q_counts.astype(float) + 1e-6

    p /= p.sum()
    q /= q.sum()

    return entropy(p, q)  # KL(P || Q)

kl_values = []
for conv_idx, group in df.groupby("conversation_idx"):
    if len(group) < 2:
        continue
    kl = kl_per_conversation(group)
    kl_values.append(kl)

kl_values = np.array(kl_values)

print("\n=== Distributional Similarity (KL Divergence) ===")
if len(kl_values) > 0:
    print(f"Mean KL divergence: {kl_values.mean():.3f}")
    print(f"Std  KL divergence: {kl_values.std():.3f}")
    print(f"Num conversations used: {len(kl_values)}")
else:
    print("Not enough data to compute KL divergences.")

# ============================================================
# 4. Conversation-level correlation:
#    avg input CMI vs avg output CMI per conversation
# ============================================================

conv_stats = (
    df.groupby("conversation_idx")[["input_cmi", "output_cmi"]]
      .mean()
      .rename(columns={"input_cmi": "mean_input_cmi", "output_cmi": "mean_output_cmi"})
      .reset_index()
)

if len(conv_stats) > 1:
    conv_pearson, _ = pearsonr(conv_stats["mean_input_cmi"], conv_stats["mean_output_cmi"])
    conv_spearman, _ = spearmanr(conv_stats["mean_input_cmi"], conv_stats["mean_output_cmi"])
else:
    conv_pearson, conv_spearman = float("nan"), float("nan")

print("\n=== Conversation-level CMI Correlation ===")
print(f"Conversation-level Pearson r = {conv_pearson:.3f}")
print(f"Conversation-level Spearman ρ = {conv_spearman:.3f}")


if len(df) > 1:
    turn_pearson, _ = pearsonr(df["input_cmi"], df["output_cmi"])
    turn_spearman, _ = spearmanr(df["input_cmi"], df["output_cmi"])
else:
    turn_pearson, turn_spearman = float("nan"), float("nan")

print("\n=== Per-turn CMI Correlation ===")
print(f"Per-turn Pearson r = {turn_pearson:.3f}")
print(f"Per-turn Spearman ρ = {turn_spearman:.3f}")

print("\nDone.")
