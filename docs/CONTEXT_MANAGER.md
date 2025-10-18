# Context Manager Integration

## Overview

The `ContextManager` class has been integrated into `AnthropicLLM` to automatically manage conversation context and prevent exceeding token limits. This ensures your conversations stay within the model's context window while preserving the most important information.

## Features

- **Automatic Context Management**: Automatically truncates messages when approaching token limits
- **Multiple Strategies**: Choose from different truncation strategies based on your needs
- **Statistics Tracking**: Monitor token usage and truncation statistics
- **Seamless Integration**: Works with all LLM methods (call, stream, call_with_tools, etc.)

## Initialization

```python
from dinnovos.llms.anthropic import AnthropicLLM

llm = AnthropicLLM(
    api_key="your-api-key",
    model="claude-sonnet-4-5-20250929",
    max_tokens=100000,          # Maximum context window (default: 100000)
    context_strategy="smart"    # Truncation strategy (default: "smart")
)
```

### Parameters

- **max_tokens** (int): Maximum tokens allowed in context. Leave room for the model's response.
- **context_strategy** (str): Strategy for truncation when context exceeds limits:
  - `"fifo"`: First In, First Out - keeps system messages and most recent messages
  - `"smart"`: Prioritizes system messages, tool calls, first few messages, and most recent messages
  - `"summary"`: Summarizes old messages (requires summary_callback)

## Usage

### Basic Call with Context Management

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    # ... many more messages ...
]

# Context management is enabled by default
response = llm.call(
    messages=messages,
    temperature=0.7,
    manage_context=True,  # Enable context management (default: True)
    verbose=True          # Show context management details (default: False)
)
```

### Streaming with Context Management

```python
for chunk in llm.stream(
    messages=messages,
    manage_context=True,
    verbose=True
):
    print(chunk, end="", flush=True)
```

### Function Calling with Context Management

```python
result = llm.call_with_tools(
    messages=messages,
    tools=tools,
    manage_context=True,
    verbose=True
)
```

### Function Execution with Context Management

```python
result = llm.call_with_function_execution(
    messages=messages,
    tools=tools,
    available_functions=available_functions,
    manage_context=True,
    verbose=True
)
```

## Context Statistics

### Get Current Statistics

```python
stats = llm.get_context_stats(messages)

print(f"Current tokens: {stats['current_tokens']}")
print(f"Max tokens: {stats['max_tokens']}")
print(f"Available tokens: {stats['available_tokens']}")
print(f"Usage: {stats['usage_percent']}%")
print(f"Messages count: {stats['messages_count']}")
print(f"Within limit: {stats['within_limit']}")
print(f"Truncated count: {stats['truncated_count']}")
print(f"Total tokens saved: {stats['total_tokens_saved']}")
```

### Reset Statistics

```python
llm.reset_context_stats()
```

## Truncation Strategies

### FIFO (First In, First Out)

Keeps system messages and the most recent messages. Oldest messages are removed first.

**Best for**: Simple conversations where recent context is most important.

```python
llm = AnthropicLLM(
    api_key=api_key,
    context_strategy="fifo"
)
```

### Smart (Recommended)

Prioritizes messages by importance:
1. System messages (always kept)
2. Tool calls and their results (high priority)
3. First 3 messages (context setting)
4. Last 5 messages (recent context)
5. Assistant responses (medium priority)
6. User messages in the middle (low priority)

**Best for**: Most use cases, especially with function calling.

```python
llm = AnthropicLLM(
    api_key=api_key,
    context_strategy="smart"  # Default
)
```

### Summary

Summarizes old messages to preserve context while reducing tokens. Requires a summary callback function.

**Best for**: Long conversations where historical context matters.

```python
def summarize_messages(messages):
    # Your summarization logic here
    # Could use another LLM call to generate summary
    return "Summary of conversation..."

# Note: Summary callback is not yet implemented in AnthropicLLM
# This is a placeholder for future enhancement
```

## Token Counting

The context manager uses a simple heuristic for token counting:
- Approximately 4 characters per token
- Accounts for message structure and metadata
- Handles both string and structured content (like tool calls)

For more accurate token counting, you could integrate a tokenizer library.

## Verbose Mode

Enable verbose mode to see detailed information about context management:

```python
response = llm.call(
    messages=messages,
    manage_context=True,
    verbose=True
)
```

Output example:
```
⚠️  Context limit reached: 105000/96000 tokens
📉 Applying 'smart' strategy...
✅ Context managed: 85000 tokens (saved 20000)
```

## Disabling Context Management

If you want to disable context management for a specific call:

```python
response = llm.call(
    messages=messages,
    manage_context=False  # Disable context management
)
```

## Best Practices

1. **Monitor Usage**: Regularly check context statistics to understand your usage patterns
2. **Choose the Right Strategy**: Use "smart" for most cases, "fifo" for simple conversations
3. **Enable Verbose Mode During Development**: Helps understand when and how truncation occurs
4. **Set Appropriate max_tokens**: Leave enough room for the model's response (default reserve: 4096 tokens)
5. **Keep System Messages Concise**: System messages are always preserved, so keep them focused

## Example: Long Conversation

```python
from dinnovos.llms.anthropic import AnthropicLLM

llm = AnthropicLLM(
    api_key="your-api-key",
    max_tokens=100000,
    context_strategy="smart"
)

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."}
]

# Simulate a long conversation
for i in range(100):
    messages.append({
        "role": "user",
        "content": f"Question {i}: Tell me about Python feature X"
    })
    
    # Context is automatically managed
    response = llm.call(messages, verbose=False)
    
    messages.append({
        "role": "assistant",
        "content": response
    })
    
    # Check if truncation occurred
    stats = llm.get_context_stats(messages)
    if stats['truncated_count'] > 0:
        print(f"Truncation occurred at message {i}")
        print(f"Tokens saved: {stats['total_tokens_saved']}")
```

## Integration with Other LLMs

The `ContextManager` is designed to be LLM-agnostic. You can integrate it with other LLM classes (OpenAI, Google, etc.) by following the same pattern used in `AnthropicLLM`.

## Future Enhancements

- Summary callback implementation for the "summary" strategy
- More accurate token counting using official tokenizers
- Custom truncation strategies
- Context window optimization for specific model versions
- Automatic strategy selection based on conversation type
