"""Anthropic LLM interface for Dinnovos Agent"""

from typing import List, Dict
from .base import BaseLLM


class AnthropicLLM(BaseLLM):
    """Interface for Anthropic models (Claude)"""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929"):
        super().__init__(api_key, model)
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Install package: pip install anthropic")
    
    def call(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Calls Anthropic API"""
        try:
            # Anthropic requires separating the system message
            system_message = ""
            formatted_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=temperature,
                system=system_message if system_message else None,
                messages=formatted_messages
            )
            
            return response.content[0].text
        except Exception as e:
            return f"Error in Anthropic: {str(e)}"