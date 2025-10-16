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
    
    def chat(
        self, 
        content: str = None, 
        temperature: float = 0.7,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Sends a message to the agent and gets a response.
        
        Args:
            user_message: User's message
            temperature: Temperature for generation
            history: Optional list of previous messages to use as context.
                     Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
                     If provided, these messages will be used instead of the internal history.
        
        Returns:
            Agent's response
        """
        # If history is provided, use it instead of internal messages
        if history is not None:
            # Build messages with system prompt + provided history + new user message
            messages_to_send = [
                {"role": "system", "content": self.system_prompt}
            ] + history
            
            # Add user message if provided
            if content is not None:
                messages_to_send.append({"role": "user", "content": content})
        else:
            # Use internal message history
            self.messages.append({"role": "user", "content": content})
            messages_to_send = self.messages
        
        # Get LLM response
        response = self.llm.call(messages_to_send, temperature=temperature)
        
        # Only update internal history if no external history was provided
        if history is None:
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