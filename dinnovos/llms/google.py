"""Google LLM interface for Dinnovos Agent"""

from typing import List, Dict
from .base import BaseLLM


class GoogleLLM(BaseLLM):
    """Interface for Google models (Gemini)"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        super().__init__(api_key, model)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("Install package: pip install google-generativeai")
    
    def call(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Calls Google Gemini API"""
        try:
            # Gemini uses a different format
            # Convert messages to Gemini format
            chat_history = []
            
            for msg in messages[:-1]:  # All except the last one
                role = "user" if msg["role"] in ["user", "system"] else "model"
                chat_history.append({
                    "role": role,
                    "parts": [msg["content"]]
                })
            
            # Start chat with history
            chat = self.client.start_chat(history=chat_history)
            
            # Send the last message
            last_message = messages[-1]["content"]
            response = chat.send_message(
                last_message,
                generation_config={"temperature": temperature}
            )
            
            return response.text
        except Exception as e:
            return f"Error in Google: {str(e)}"