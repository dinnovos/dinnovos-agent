"""Base LLM interface for Dinnovos Agent"""

from typing import List, Dict, Iterator
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for all LLMs"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
    
    @abstractmethod
    def call(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Abstract method that all interfaces must implement.
        
        Args:
            messages: List of messages in format [{"role": "user/assistant/system", "content": "..."}]
            temperature: Temperature for generation (0-1)
        
        Returns:
            LLM response as string
        """
        pass
    
    @abstractmethod
    def stream(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Iterator[str]:
        """
        Abstract method for streaming responses.
        
        Args:
            messages: List of messages in format [{"role": "user/assistant/system", "content": "..."}]
            temperature: Temperature for generation (0-1)
        
        Yields:
            Chunks of the LLM response as strings
        """
        pass