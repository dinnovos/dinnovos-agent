"""OpenAI LLM interface for Dinnovos Agent"""

from typing import List, Dict
from .base import BaseLLM


class OpenAILLM(BaseLLM):
    """Interface for OpenAI models (GPT-4, GPT-3.5, etc.)"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__(api_key, model)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Install package: pip install openai")
    
    def call(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Calls OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in OpenAI: {str(e)}"