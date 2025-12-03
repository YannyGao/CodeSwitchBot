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
