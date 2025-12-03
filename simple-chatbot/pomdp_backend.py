"""
POMDP Backend for Adaptive Multilingual Chatbot
AA228/CS238 Project

This module implements a lightweight POMDP for learning user language preferences
and choosing appropriate response modes (English, Mandarin, mixed, or translate).

The POMDP tracks belief over hidden user preference states and selects actions
that maximize expected reward based on matching the user's code-switching style.
"""

import numpy as np
import re
from typing import Tuple, Dict, List

# =============================================================================
# POMDP State and Action Definitions
# =============================================================================

# Hidden states representing user's language preference
STATES = ["EN_DOMINANT", "MIXED", "ZH_DOMINANT"]
N_STATES = len(STATES)

# Available actions for response generation
ACTIONS = [
    "RESPOND_EN",      # Reply fully in English
    "RESPOND_ZH",      # Reply fully in Mandarin
    "MIXED_RESPONSE",  # Code-switch similar to user
    "TRANSLATE_QUERY"  # Clarify/translate before responding
]
N_ACTIONS = len(ACTIONS)


class UserPreferencePOMDP:
    """
    POMDP model for inferring user language preferences.
    
    States: EN_DOMINANT, MIXED, ZH_DOMINANT (hidden)
    Observations: Language mix features from user text
    Actions: Response language choices
    
    This is an explicit POMDP with:
    - Transition model: User preference tends to persist
    - Observation model: P(observed mix | true preference)
    - Reward: Matching user preference yields higher reward
    """
    
    def __init__(self):
        """Initialize POMDP with uniform prior belief."""
        
        # Belief distribution over hidden states
        # Start with slight English bias (common in EN-ZH conversations)
        self.belief = np.array([0.5, 0.3, 0.2])
        
        # -----------------------------------------------------------------
        # Transition Model: P(s' | s, a)
        # User preference mostly persists; small probability of shifting
        # Actions can slightly influence transitions (e.g., using ZH may
        # encourage user to use more ZH)
        # -----------------------------------------------------------------
        # Base transition (preference tends to persist)
        self.T_base = np.array([
            #  EN_DOM  MIXED  ZH_DOM  (next state)
            [   0.85,  0.12,  0.03],  # from EN_DOMINANT
            [   0.15,  0.70,  0.15],  # from MIXED
            [   0.03,  0.12,  0.85],  # from ZH_DOMINANT
        ])
        
        # -----------------------------------------------------------------
        # Observation Model: P(o | s)
        # o = discretized observation bucket based on language mix
        # Observation buckets: EN_HEAVY, BALANCED, ZH_HEAVY
        # -----------------------------------------------------------------
        self.O = np.array([
            # P(obs | state)     EN_HEAVY  BALANCED  ZH_HEAVY
            # EN_DOMINANT:
            [   0.75,            0.20,     0.05],
            # MIXED:
            [   0.20,            0.60,     0.20],
            # ZH_DOMINANT:
            [   0.05,            0.20,     0.75],
        ])
        
        # -----------------------------------------------------------------
        # Reward Model: R(s, a)
        # Reward for taking action a when true state is s
        # Higher reward for matching user preference
        # -----------------------------------------------------------------
        self.R = np.array([
            #              RESPOND_EN  RESPOND_ZH  MIXED  TRANSLATE
            # EN_DOMINANT:
            [               1.0,       -0.5,       0.3,   0.0],
            # MIXED:
            [               0.2,        0.2,       1.0,   0.5],
            # ZH_DOMINANT:
            [              -0.5,        1.0,       0.3,   0.0],
        ])
    
    def get_belief(self) -> np.ndarray:
        """Return current belief distribution."""
        return self.belief.copy()
    
    def get_belief_dict(self) -> Dict[str, float]:
        """Return belief as labeled dictionary for display."""
        return {STATES[i]: float(self.belief[i]) for i in range(N_STATES)}
    
    def update_belief(self, obs_bucket: int) -> np.ndarray:
        """
        Bayesian belief update given observation.
        
        b'(s') ∝ P(o|s') * Σ_s P(s'|s) * b(s)
        
        Args:
            obs_bucket: Discretized observation (0=EN_HEAVY, 1=BALANCED, 2=ZH_HEAVY)
        
        Returns:
            Updated belief distribution
        """
        # Prediction step: b_pred(s') = Σ_s P(s'|s) * b(s)
        b_pred = self.T_base.T @ self.belief
        
        # Update step: b'(s') ∝ P(o|s') * b_pred(s')
        likelihood = self.O[:, obs_bucket]
        b_new = likelihood * b_pred
        
        # Normalize
        b_new = b_new / (b_new.sum() + 1e-10)
        
        self.belief = b_new
        return self.belief.copy()
    
    def expected_reward(self, action_idx: int) -> float:
        """
        Compute expected reward for an action under current belief.
        
        E[R(a)] = Σ_s b(s) * R(s, a)
        """
        return float(self.belief @ self.R[:, action_idx])
    
    def reset(self):
        """Reset belief to prior."""
        self.belief = np.array([0.5, 0.3, 0.2])


class POMDPLanguagePolicy:
    """
    Policy layer that extracts observations from user text,
    updates POMDP belief, and selects response actions.
    
    This integrates with the chatbot to make language decisions.
    """
    
    def __init__(self):
        """Initialize policy with POMDP model."""
        self.pomdp = UserPreferencePOMDP()
        self.last_action = None
        self.last_action_name = None
        self.last_observation = None
        self.last_cmi = 0.0
    
    def compute_observation(self, text: str) -> Tuple[int, Dict]:
        """
        Extract language features from user text and discretize to observation bucket.
        
        Uses simple regex-based language detection:
        - Chinese: Unicode Han characters
        - English: Latin alphabet
        
        Returns:
            obs_bucket: 0 (EN_HEAVY), 1 (BALANCED), 2 (ZH_HEAVY)
            features: Dictionary of extracted features
        """
        # Count Chinese characters (Han script)
        zh_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        
        # Count English words (sequences of Latin letters)
        en_words = len(re.findall(r'[a-zA-Z]+', text))
        en_chars = sum(len(w) for w in re.findall(r'[a-zA-Z]+', text))
        
        total_chars = zh_chars + en_chars
        
        if total_chars == 0:
            # No language content detected (numbers, punctuation only)
            zh_ratio = 0.5  # Neutral
        else:
            zh_ratio = zh_chars / total_chars
        
        # Compute Code-Mixing Index (CMI)
        # CMI = 100 * (1 - max(p_en, p_zh))
        # Range: 0 (monolingual) to 50 (perfectly balanced)
        if total_chars > 0:
            cmi = 100 * (1 - max(zh_ratio, 1 - zh_ratio))
        else:
            cmi = 0.0
        
        self.last_cmi = cmi
        
        # Discretize to observation buckets
        # EN_HEAVY: >70% English (zh_ratio < 0.3)
        # BALANCED: 30-70% each
        # ZH_HEAVY: >70% Chinese (zh_ratio > 0.7)
        if zh_ratio < 0.3:
            obs_bucket = 0  # EN_HEAVY
        elif zh_ratio > 0.7:
            obs_bucket = 2  # ZH_HEAVY
        else:
            obs_bucket = 1  # BALANCED
        
        features = {
            "zh_chars": zh_chars,
            "en_chars": en_chars,
            "zh_ratio": zh_ratio,
            "cmi": cmi,
            "obs_bucket": ["EN_HEAVY", "BALANCED", "ZH_HEAVY"][obs_bucket]
        }
        
        return obs_bucket, features
    
    def select_action(self, user_message: str) -> Tuple[str, Dict]:
        """
        Main policy function: observe, update belief, select action.
        
        Args:
            user_message: Raw user input text
        
        Returns:
            action_name: Selected action string
            info: Dictionary with belief, features, expected rewards
        """
        # 1. Extract observation from user message
        obs_bucket, features = self.compute_observation(user_message)
        self.last_observation = features
        
        # 2. Update POMDP belief
        belief_before = self.pomdp.get_belief_dict()
        self.pomdp.update_belief(obs_bucket)
        belief_after = self.pomdp.get_belief_dict()
        
        # 3. Compute expected rewards for each action
        expected_rewards = {
            ACTIONS[i]: self.pomdp.expected_reward(i) 
            for i in range(N_ACTIONS)
        }
        
        # 4. Select action with highest expected reward (greedy policy)
        best_action_idx = np.argmax([self.pomdp.expected_reward(i) for i in range(N_ACTIONS)])
        action_name = ACTIONS[best_action_idx]
        
        self.last_action = best_action_idx
        self.last_action_name = action_name
        
        # Compile info for logging/display
        info = {
            "observation": features,
            "belief_before": belief_before,
            "belief_after": belief_after,
            "expected_rewards": expected_rewards,
            "selected_action": action_name,
            "cmi": features["cmi"]
        }
        
        return action_name, info
    
    def get_system_prompt(self, action: str, user_cmi: float = 0.0) -> str:
        """
        Generate system prompt to condition LLM based on POMDP action.
        
        Args:
            action: Selected POMDP action
            user_cmi: Code-mixing index of user's message (for MIXED mode)
        
        Returns:
            System prompt string
        """
        prompts = {
            "RESPOND_EN": (
                "You are a helpful assistant. Respond ONLY in English. "
                "Do not use any Chinese characters in your response."
            ),
            "RESPOND_ZH": (
                "你是一个有帮助的助手。请只用中文回答。"
                "不要在回答中使用任何英文。"
            ),
            "MIXED_RESPONSE": (
                f"You are a helpful bilingual assistant chatting with a user who code-switches "
                f"between English and Mandarin. The user's message has a code-mixing index of {user_cmi:.1f}. "
                f"Mirror their style: mix English and Chinese naturally in your response. "
                f"Use both languages fluidly as a bilingual speaker would."
            ),
            "TRANSLATE_QUERY": (
                "You are a helpful bilingual assistant. The user's language preference is unclear. "
                "First, briefly clarify or paraphrase their message in both English and Chinese, "
                "then provide a helpful response in both languages."
            )
        }
        return prompts.get(action, prompts["RESPOND_EN"])
    
    def get_belief(self) -> Dict[str, float]:
        """Return current belief distribution."""
        return self.pomdp.get_belief_dict()
    
    def get_last_action(self) -> str:
        """Return last selected action."""
        return self.last_action_name or "None"
    
    def get_last_cmi(self) -> float:
        """Return CMI of last user message."""
        return self.last_cmi
    
    def reset(self):
        """Reset policy and POMDP belief."""
        self.pomdp.reset()
        self.last_action = None
        self.last_action_name = None
        self.last_observation = None
        self.last_cmi = 0.0


# =============================================================================
# Sanity Check Tests
# =============================================================================

def test_pomdp():
    """Test POMDP belief updates and policy behavior."""
    print("=" * 60)
    print("POMDP Sanity Checks")
    print("=" * 60)
    
    policy = POMDPLanguagePolicy()
    
    # Test 1: English-dominant input
    print("\n[Test 1] English input: 'Hello, how are you today?'")
    action, info = policy.select_action("Hello, how are you today?")
    print(f"  Observation: {info['observation']['obs_bucket']}")
    print(f"  Belief: {info['belief_after']}")
    print(f"  Action: {action}")
    assert action == "RESPOND_EN", "Expected English response for English input"
    
    # Test 2: Chinese-dominant input
    print("\n[Test 2] Chinese input: '你好，今天怎么样？'")
    action, info = policy.select_action("你好，今天怎么样？")
    print(f"  Observation: {info['observation']['obs_bucket']}")
    print(f"  Belief: {info['belief_after']}")
    print(f"  Action: {action}")
    # After one ZH input, might still not switch fully
    
    # Test 3: Multiple Chinese inputs should shift belief
    print("\n[Test 3] Multiple Chinese inputs...")
    for i in range(3):
        action, info = policy.select_action("这个问题很有趣，我想了解更多")
    print(f"  Final Belief: {info['belief_after']}")
    print(f"  Final Action: {action}")
    assert info['belief_after']['ZH_DOMINANT'] > 0.5, "Expected ZH_DOMINANT belief after multiple ZH inputs"
    
    # Test 4: Mixed code-switching input
    policy.reset()
    print("\n[Test 4] Mixed input: 'Hey 你知道这个 project 怎么做吗？'")
    action, info = policy.select_action("Hey 你知道这个 project 怎么做吗？")
    print(f"  Observation: {info['observation']['obs_bucket']}")
    print(f"  CMI: {info['cmi']:.2f}")
    print(f"  Belief: {info['belief_after']}")
    print(f"  Action: {action}")
    
    # Test 5: System prompt generation
    print("\n[Test 5] System prompts:")
    for act in ACTIONS:
        prompt = policy.get_system_prompt(act, user_cmi=25.0)
        print(f"  {act}: {prompt[:60]}...")
    
    print("\n" + "=" * 60)
    print("All sanity checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_pomdp()

