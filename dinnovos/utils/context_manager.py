"""Context Manager for LLM conversations"""

from typing import List, Dict, Any, Optional, Callable
import json


class ContextManager:
    """Manages context window and token limits for LLM conversations"""
    
    def __init__(
        self,
        max_tokens: int = 100000,
        strategy: str = "smart",
        reserve_tokens: int = 4096,
        summary_callback: Optional[Callable] = None
    ):
        """
        Initialize the context manager.
        
        Args:
            max_tokens: Maximum tokens allowed in context (leave room for response)
            strategy: Strategy for truncation - "fifo", "summary", or "smart"
            reserve_tokens: Tokens to reserve for the LLM response
            summary_callback: Optional callback function to generate summaries
        """
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.reserve_tokens = reserve_tokens
        self.summary_callback = summary_callback
        self.truncated_count = 0
        self.total_tokens_saved = 0
    
    def manage(self, messages: List[Dict[str, Any]], verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Manage context by truncating if necessary.
        
        Args:
            messages: List of conversation messages
            verbose: If True, print debug information
        
        Returns:
            Managed list of messages within token limit
        """
        available_tokens = self.max_tokens - self.reserve_tokens
        current_tokens = self.count_tokens(messages)
        
        if current_tokens <= available_tokens:
            return messages
        
        if verbose:
            print(f"⚠️  Context limit reached: {current_tokens}/{available_tokens} tokens")
            print(f"📉 Applying '{self.strategy}' strategy...")
        
        # Apply truncation strategy
        if self.strategy == "fifo":
            result = self._truncate_fifo(messages, available_tokens)
        elif self.strategy == "summary":
            result = self._truncate_with_summary(messages, available_tokens, verbose)
        elif self.strategy == "smart":
            result = self._truncate_smart(messages, available_tokens)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Update statistics
        self.truncated_count += 1
        tokens_saved = current_tokens - self.count_tokens(result)
        self.total_tokens_saved += tokens_saved
        
        if verbose:
            final_tokens = self.count_tokens(result)
            print(f"✅ Context managed: {final_tokens} tokens (saved {tokens_saved})")
        
        return result
    
    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count approximate tokens in messages.
        Uses a simple heuristic: ~4 characters per token.
        
        Args:
            messages: List of messages to count
        
        Returns:
            Approximate token count
        """
        total = 0
        
        for msg in messages:
            # Count content
            content = msg.get("content")
            if content:
                if isinstance(content, str):
                    total += len(content) // 4
                elif isinstance(content, list):
                    # Handle structured content (like Anthropic format)
                    for item in content:
                        if isinstance(item, dict):
                            total += len(str(item)) // 4
                        else:
                            total += len(str(item)) // 4
            
            # Count tool calls
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    total += len(json.dumps(tc)) // 4
            
            # Count role and other metadata (small overhead)
            total += 10
        
        return total
    
    def _truncate_fifo(self, messages: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """
        First In, First Out truncation.
        Keeps system messages and most recent messages.
        """
        result = []
        current_tokens = 0
        
        # Always keep system messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        result.extend(system_msgs)
        current_tokens += self.count_tokens(system_msgs)
        
        # Add messages from most recent, working backwards
        non_system = [m for m in messages if m.get("role") != "system"]
        
        for msg in reversed(non_system):
            msg_tokens = self.count_tokens([msg])
            if current_tokens + msg_tokens > max_tokens:
                break
            
            result.insert(len(system_msgs), msg)
            current_tokens += msg_tokens
        
        return result
    
    def _truncate_smart(self, messages: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """
        Smart truncation that prioritizes:
        - System messages
        - Tool calls and their results
        - First few messages (context setting)
        - Most recent messages
        """
        if self.count_tokens(messages) <= max_tokens:
            return messages
        
        # Categorize messages by priority
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for i, msg in enumerate(messages):
            role = msg.get("role")
            
            # High priority
            if (role == "system" or 
                msg.get("tool_calls") or 
                role == "tool" or
                i < 3 or  # First 3 messages
                i >= len(messages) - 5):  # Last 5 messages
                high_priority.append((i, msg))
            # Medium priority (assistant responses)
            elif role == "assistant":
                medium_priority.append((i, msg))
            # Low priority (user messages in the middle)
            else:
                low_priority.append((i, msg))
        
        # Build result starting with high priority
        result = []
        current_tokens = 0
        
        # Add all high priority messages
        for _, msg in high_priority:
            result.append(msg)
            current_tokens += self.count_tokens([msg])
        
        # Add medium priority if space available
        for _, msg in medium_priority:
            msg_tokens = self.count_tokens([msg])
            if current_tokens + msg_tokens <= max_tokens:
                result.append(msg)
                current_tokens += msg_tokens
        
        # Add low priority if space available
        for _, msg in low_priority:
            msg_tokens = self.count_tokens([msg])
            if current_tokens + msg_tokens <= max_tokens:
                result.append(msg)
                current_tokens += msg_tokens
        
        # Sort by original order
        result.sort(key=lambda m: messages.index(m))
        
        return result
    
    def _truncate_with_summary(self, messages: List[Dict[str, Any]], max_tokens: int, verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Truncate by summarizing old messages.
        Requires a summary_callback function.
        """
        if not self.summary_callback:
            # Fallback to smart truncation if no callback
            if verbose:
                print("⚠️  No summary callback provided, using smart truncation")
            return self._truncate_smart(messages, max_tokens)
        
        # Separate messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        
        # Keep last N messages
        keep_recent = 10
        old_msgs = non_system[:-keep_recent] if len(non_system) > keep_recent else []
        recent_msgs = non_system[-keep_recent:] if len(non_system) > keep_recent else non_system
        
        if not old_msgs:
            # Nothing to summarize, use smart truncation
            return self._truncate_smart(messages, max_tokens)
        
        # Generate summary using callback
        try:
            summary_text = self.summary_callback(old_msgs)
            
            # Build result with summary
            result = system_msgs + [
                {
                    "role": "system",
                    "content": f"[CONVERSATION SUMMARY]\n{summary_text}\n[END SUMMARY]"
                }
            ] + recent_msgs
            
            # Check if still too large
            if self.count_tokens(result) > max_tokens:
                if verbose:
                    print("⚠️  Summary still too large, applying smart truncation")
                return self._truncate_smart(result, max_tokens)
            
            return result
            
        except Exception as e:
            if verbose:
                print(f"⚠️  Error generating summary: {e}")
            return self._truncate_smart(messages, max_tokens)
    
    def get_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about current context usage.
        
        Args:
            messages: Current message list
        
        Returns:
            Dictionary with usage statistics
        """
        current_tokens = self.count_tokens(messages)
        available_tokens = self.max_tokens - self.reserve_tokens
        
        return {
            "current_tokens": current_tokens,
            "max_tokens": self.max_tokens,
            "available_tokens": available_tokens,
            "reserve_tokens": self.reserve_tokens,
            "usage_percent": round((current_tokens / available_tokens) * 100, 2),
            "messages_count": len(messages),
            "truncated_count": self.truncated_count,
            "total_tokens_saved": self.total_tokens_saved,
            "within_limit": current_tokens <= available_tokens
        }
    
    def reset_stats(self):
        """Reset truncation statistics"""
        self.truncated_count = 0
        self.total_tokens_saved = 0