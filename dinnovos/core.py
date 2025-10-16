"""Core Dinnovos Agent implementation"""

from typing import List, Dict, Optional
from .llms.base import BaseLLM


class Agent:
    """
    Agent - Agile and intelligent conversational agent

    An agent that can use any LLM (OpenAI, Anthropic, Google)
    and maintains conversations with context memory.
    """
    
    def __init__(
        self, 
        llm: BaseLLM,
        system_prompt: Optional[str] = None,
        max_history: int = 10
    ):
        """
        Args:
            llm: LLM interface to use (OpenAI, Anthropic or Google)
            system_prompt: System instructions for the agent
            max_history: Maximum number of messages to keep in memory
        """
        self.llm = llm
        self.system_prompt = system_prompt or "You are a helpful and concise assistant."
        self.max_history = max_history
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def chat(self, user_message: str, temperature: float = 0.7) -> str:
        """
        Sends a message to the agent and gets a response.
        
        Args:
            user_message: User's message
            temperature: Temperature for generation
        
        Returns:
            Agent's response
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_message})
        
        # Get LLM response
        response = self.llm.call(self.messages, temperature=temperature)
        
        # Add assistant response
        self.messages.append({"role": "assistant", "content": response})
        
        # Keep only the last N messages (+ system prompt)
        if len(self.messages) > self.max_history + 1:
            # Keep system prompt + last max_history messages
            self.messages = [self.messages[0]] + self.messages[-(self.max_history):]
        
        return response
    
    def reset(self):
        """Resets the conversation"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
    
    def get_history(self) -> List[Dict[str, str]]:
        """Gets message history"""
        return self.messages.copy()
    
    def set_system_prompt(self, new_prompt: str):
        """Changes the system prompt and resets the conversation"""
        self.system_prompt = new_prompt
        self.reset()