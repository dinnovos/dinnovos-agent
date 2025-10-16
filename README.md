# 🦖 Dinnovos Agent

**Agile AI Agents with Multi-LLM Support**

Dinnovos Agent is a lightweight Python framework for building AI agents that can seamlessly switch between different Large Language Models (OpenAI, Anthropic, Google).

## Features

- 🔄 **Multi-LLM Support**: OpenAI (GPT), Anthropic (Claude), Google (Gemini)
- 🎯 **Simple API**: Intuitive interface for building conversational agents
- 💾 **Context Memory**: Automatic conversation history management
- 🔌 **Extensible**: Easy to add new LLM providers
- 🪶 **Lightweight**: Minimal dependencies, maximum flexibility

## Installation

### Basic Installation
```bash
pip install dinnovos-agent
```

### With specific LLM support
```bash
# For OpenAI only
pip install dinnovos-agent[openai]

# For Anthropic only
pip install dinnovos-agent[anthropic]

# For Google only
pip install dinnovos-agent[google]

# For all LLMs
pip install dinnovos-agent[all]
```

### For development
```bash
pip install dinnovos-agent[dev]
```

## Quick Start

```python
from dinnovos import Dinnovos, OpenAILLM

# Create an LLM interface
llm = OpenAILLM(api_key="your-api-key", model="gpt-4")

# Create a Dinnovos agent
dinnovos = Dinnovos(
    llm=llm,
    system_prompt="You are a helpful assistant."
)

# Chat with your agent
response = dinnovos.chat("Hello! What can you do?")
print(response)
```

## Examples

### Using Different LLMs

```python
from dinnovos import Dinnovos, OpenAILLM, AnthropicLLM, GoogleLLM

# OpenAI
openai_llm = OpenAILLM(api_key="sk-...", model="gpt-4")
dinnovos_gpt = Dinnovos(llm=openai_llm)

# Anthropic Claude
anthropic_llm = AnthropicLLM(api_key="sk-ant-...", model="claude-sonnet-4-5-20250929")
dinnovos_claude = Dinnovos(llm=anthropic_llm)

# Google Gemini
google_llm = GoogleLLM(api_key="...", model="gemini-1.5-pro")
dinnovos_gemini = Dinnovos(llm=google_llm)
```

### Custom System Prompt

```python
dinnovos = Dinnovos(
    llm=llm,
    system_prompt="You are an expert Python programmer.",
    max_history=20  # Keep last 20 messages
)

response = dinnovos.chat("Explain decorators in Python")
```

### Managing Conversation

```python
# Get conversation history
history = dinnovos.get_history()

# Reset conversation
dinnovos.reset()

# Change system prompt
dinnovos.set_system_prompt("You are now a math tutor.")
```

## API Reference

### Dinnovos Class

```python
Dinnovos(llm: BaseLLM, system_prompt: str = None, max_history: int = 10)
```

**Methods:**
- `chat(message: str, temperature: float = 0.7) -> str`: Send a message and get response
- `reset()`: Clear conversation history
- `get_history() -> List[Dict]`: Get conversation history
- `set_system_prompt(prompt: str)`: Change system prompt and reset

### LLM Interfaces

```python
OpenAILLM(api_key: str, model: str = "gpt-4")
AnthropicLLM(api_key: str, model: str = "claude-sonnet-4-5-20250929")
GoogleLLM(api_key: str, model: str = "gemini-1.5-pro")
```

## Requirements

- Python 3.8+
- Optional: `openai`, `anthropic`, `google-generativeai` (based on which LLMs you use)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Links

- [Documentation](https://github.com/yourusername/dinnovos-agent/docs)
- [Issue Tracker](https://github.com/yourusername/dinnovos-agent/issues)
- [Source Code](https://github.com/yourusername/dinnovos-agent)

## Support

If you encounter any issues or have questions, please file an issue on GitHub.
'''