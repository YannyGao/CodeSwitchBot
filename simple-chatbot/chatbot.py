"""
Chatbot Backend with POMDP-Based Language Selection
AA228/CS238 Project

This module provides chatbot classes that use LLMs for generation,
with language decisions controlled by a POMDP policy.

Supports:
- Together API (Llama 3.1 8B Instruct) - cloud-based, high quality
- Local Qwen model - free, runs locally (optional, requires transformers)

The POMDP policy observes user language patterns and selects
appropriate response modes (English, Chinese, mixed, translate).
"""

import os
from openai import OpenAI
from pomdp_backend import POMDPLanguagePolicy

# third-party imports
try:
    import requests
except Exception as e:
    # make the error more actionable
    raise ImportError(
        "Missing dependency 'requests'. Install with: pip install requests\n"
        f"Original error: {e}"
    )
from dotenv import load_dotenv
import os

try:
    from mistralai import Mistral
except Exception as e:
    raise ImportError(
        "Missing dependency 'mistralai'. Install with: pip install mistralai\n"
        f"Original error: {e}"
    )

load_dotenv()  # reads .env and sets environment variables

# check that token is loaded
print(os.environ.get("TOGETHER_API_KEY"))
print(os.environ.get("HUGGINGFACE_HUB_TOKEN"))


class TogetherChat:
    """
    Together API chatbot with POMDP-controlled language selection.
    
    Uses Llama 3.1 8B Instruct via Together API.
    The POMDP policy determines response language based on user patterns.
    """
    
    def __init__(self, use_pomdp: bool = True):
        """
        Initialize Together API client and POMDP policy.
        
        Args:
            use_pomdp: Whether to use POMDP for language decisions
        """
        # Initialize Together API client (OpenAI-compatible)
        self.client = OpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1"
        )
        self.model = "meta-llama/Llama-3.2-3B-Instruct-Turbo"  # Faster than 8B
        
        # Conversation history
        self.history = []
        
        # POMDP policy for language selection
        self.use_pomdp = use_pomdp
        self.policy = POMDPLanguagePolicy() if use_pomdp else None
        
        # Store last POMDP info for UI display
        self.last_pomdp_info = None
    
    def chat(self, message: str) -> str:
        """
        Process user message and generate response.
        
        If POMDP is enabled:
        1. Policy observes user message and updates belief
        2. Policy selects action (language mode)
        3. System prompt conditions LLM accordingly
        4. LLM generates response in chosen mode
        
        Args:
            message: User's input message
        
        Returns:
            Assistant's response
        """
        # POMDP action selection
        if self.use_pomdp and self.policy:
            action, info = self.policy.select_action(message)
            self.last_pomdp_info = info
            
            # Get system prompt based on action
            system_prompt = self.policy.get_system_prompt(action, info["cmi"])
        else:
            system_prompt = "You are a helpful bilingual assistant."
            self.last_pomdp_info = None
        
        # Build messages with system prompt
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        messages.extend(self.history)
        
        # Add current user message
        messages.append({"role": "user", "content": message})
        
        # Call Together API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        reply = response.choices[0].message.content
        
        # Update history
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": reply})
        
        return reply
    
    def get_belief(self):
        """Return current POMDP belief distribution."""
        if self.policy:
            return self.policy.get_belief()
        return None
    
    def get_last_action(self):
        """Return last POMDP action."""
        if self.policy:
            return self.policy.get_last_action()
        return None
    
    def get_last_cmi(self):
        """Return CMI of last user message."""
        if self.policy:
            return self.policy.get_last_cmi()
        return 0.0
    
    def get_last_pomdp_info(self):
        """Return full POMDP info from last turn."""
        return self.last_pomdp_info
    
    def reset(self):
        """Reset conversation history and POMDP belief."""
        self.history = []
        if self.policy:
            self.policy.reset()
        self.last_pomdp_info = None


class MistralChat:
    """
    Mistral model via official mistralai SDK (direct to Mistral).
    Integrates with the POMDP policy the same way as TogetherChat/LocalChat.
    """
    def __init__(self, model: str = "mistral-medium-latest", use_pomdp: bool = True):
        """
        Args:
            model: Mistral model id (e.g. "mistral-medium-latest", "mistral-large-latest")
            use_pomdp: whether to use POMDP policy for language decisions
        """
        self.model = model
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "MISTRAL_API_KEY not set. Set it in .env or environment to use the Mistral API."
            )

        # Create client (the SDK typically expects api_key kwarg)
        # If the SDK signature changes, adjust accordingly
        self.client = Mistral(api_key=self.api_key)

        # conversation history (optional; for simple stateless calls only the current message is needed)
        self.history = []

        # POMDP policy
        self.use_pomdp = use_pomdp
        self.policy = POMDPLanguagePolicy() if use_pomdp else None
        self.last_pomdp_info = None

    def _extract_text_from_response(self, resp):
        """
        Try several plausible shapes returned by mistralai client.chat.complete.
        This is defensive: exact SDK return shape may be an object with attributes
        or a dict with fields; adjust if needed for your SDK version.
        """
        # If SDK returns an object with a simple 'content' or 'completion' attribute
        if hasattr(resp, "content"):
            return getattr(resp, "content")
        if hasattr(resp, "completion"):
            return getattr(resp, "completion")
        # If the SDK returns a structure with generations / choices
        try:
            # common shapes
            if isinstance(resp, dict):
                # e.g., {"completion": "..."} or {"output": "..."} or {"choices": [{"text": "..."}]}
                if "completion" in resp and isinstance(resp["completion"], str):
                    return resp["completion"]
                if "output" in resp and isinstance(resp["output"], str):
                    return resp["output"]
                if "generated_text" in resp and isinstance(resp["generated_text"], str):
                    return resp["generated_text"]
                if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                    first = resp["choices"][0]
                    if isinstance(first, dict):
                        for key in ("text", "content", "message", "generated_text"):
                            if key in first and isinstance(first[key], str):
                                return first[key]
                        # sometimes HF-like: first["message"]["content"]
                        if "message" in first and isinstance(first["message"], dict):
                            msg = first["message"]
                            if "content" in msg:
                                return msg["content"]
                # fallback: join text fields
                for v in ("text", "completion", "generated_text", "output"):
                    if v in resp and isinstance(resp[v], str):
                        return resp[v]
            # If object-like with nested structure
            if hasattr(resp, "generations") and resp.generations:
                g0 = resp.generations[0]
                if hasattr(g0, "text"):
                    return g0.text
                if isinstance(g0, dict) and "text" in g0:
                    return g0["text"]
        except Exception:
            pass

        # Last resort: stringified JSON
        try:
            return json.dumps(resp, ensure_ascii=False)
        except Exception:
            return str(resp)

    def _call_mistral(self, messages, temperature: float = 0.7, max_tokens: int = 512):
        """
        Call the Mistral client.chat.complete API.
        messages should be a list of {"role": "...", "content": "..."} dicts.
        """
        # The sample you provided used client.chat.complete(model=..., messages=[...])
        # Add other params as supported by your SDK (e.g., temperature, max_tokens)
        try:
            resp = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            # propagate useful error
            raise RuntimeError(f"Mistral API call failed: {e}")

        text = self._extract_text_from_response(resp)
        return text.strip() if isinstance(text, str) else str(text)

    def chat(self, message: str) -> str:
        """
        Process user message, optionally use POMDP to condition the system prompt,
        then send to Mistral and return the assistant text.
        """
        # 1) POMDP: select action and system prompt
        if self.use_pomdp and self.policy:
            action, info = self.policy.select_action(message)
            self.last_pomdp_info = info
            system_prompt = self.policy.get_system_prompt(action, info["cmi"])
        else:
            system_prompt = "You are a helpful bilingual assistant."
            self.last_pomdp_info = None

        # 2) Build messages for Mistral chat format
        # Mistral sample uses role/message dicts; follow same shape
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]
        # Optionally include history if you want multi-turn context
        # e.g., extend messages with previous user/assistant entries from self.history

        # 3) Call API
        reply_text = self._call_mistral(messages, temperature=0.7, max_tokens=500)

        # 4) Update history
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": reply_text})

        return reply_text

    def get_belief(self):
        return self.policy.get_belief() if self.policy else None

    def get_last_action(self):
        return self.policy.get_last_action() if self.policy else None

    def get_last_cmi(self):
        return self.policy.get_last_cmi() if self.policy else 0.0

    def get_last_pomdp_info(self):
        return self.last_pomdp_info

    def reset(self):
        self.history = []
        if self.policy:
            self.policy.reset()
        self.last_pomdp_info = None

class LocalChat:
    """
    Local Hugging Face model chatbot with POMDP-controlled language selection.
    
    Uses Qwen 2.5 1.5B Instruct by default (good for EN+ZH).
    Runs locally without API costs.
    
    Note: Requires transformers and torch to be installed correctly.
    """
    
    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct", use_pomdp: bool = True):
        """
        Initialize local model and POMDP policy.
        
        Args:
            model: HuggingFace model name
            use_pomdp: Whether to use POMDP for language decisions
        """
        # Lazy import to avoid issues if transformers has problems
        try:
            import torch
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                f"Local model requires transformers and torch. Error: {e}\n"
                "Try: pip install transformers torch\n"
                "Or use the Together API option instead."
            )
        
        # Initialize local model pipeline
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "text-generation",
            model=model,
            device=device,
            max_new_tokens=200,
            truncation=True
        )
        self.model_name = model
        
        # POMDP policy
        self.use_pomdp = use_pomdp
        self.policy = POMDPLanguagePolicy() if use_pomdp else None
        
        # Store last POMDP info
        self.last_pomdp_info = None
    
    def chat(self, message: str) -> str:
        """
        Process user message and generate response with POMDP control.
        
        Args:
            message: User's input message
        
        Returns:
            Assistant's response
        """
        # POMDP action selection
        if self.use_pomdp and self.policy:
            action, info = self.policy.select_action(message)
            self.last_pomdp_info = info
            system_prompt = self.policy.get_system_prompt(action, info["cmi"])
        else:
            system_prompt = "You are a helpful assistant."
            self.last_pomdp_info = None
        
        # Build prompt for Qwen format
        if "Qwen" in self.model_name:
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{message}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            result = self.pipe(
                prompt, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=self.pipe.tokenizer.eos_token_id
            )[0]['generated_text']
            
            # Extract assistant response
            if "<|im_start|>assistant" in result:
                response = result.split("<|im_start|>assistant")[-1].strip()
                response = response.split("<|im_end|>")[0].strip()
            else:
                response = result.replace(prompt, "").strip()
        else:
            # Generic format for other models
            prompt = f"System: {system_prompt}\nUser: {message}\nAssistant:"
            result = self.pipe(
                prompt,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.pipe.tokenizer.eos_token_id
            )[0]['generated_text']
            response = result.replace(prompt, "").strip()
        
        return response if response else "I understand. Can you tell me more?"
    
    def get_belief(self):
        """Return current POMDP belief distribution."""
        if self.policy:
            return self.policy.get_belief()
        return None
    
    def get_last_action(self):
        """Return last POMDP action."""
        if self.policy:
            return self.policy.get_last_action()
        return None
    
    def get_last_cmi(self):
        """Return CMI of last user message."""
        if self.policy:
            return self.policy.get_last_cmi()
        return 0.0
    
    def get_last_pomdp_info(self):
        """Return full POMDP info from last turn."""
        return self.last_pomdp_info
    
    def reset(self):
        """Reset POMDP belief."""
        if self.policy:
            self.policy.reset()
        self.last_pomdp_info = None
