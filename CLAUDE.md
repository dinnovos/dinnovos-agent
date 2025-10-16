# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dinnovos Agent is a lightweight Python framework for building AI agents with multi-LLM support. The package provides a unified interface for interacting with OpenAI (GPT), Anthropic (Claude), and Google (Gemini) models through a single conversational agent API.

## Development Commands

### Setup
```bash
# Create and activate virtual environment (existing: vienv/)
python -m venv vienv
vienv\Scripts\activate  # Windows
source vienv/bin/activate  # Unix/macOS

# Install for development
pip install -e .[dev]

# Install with specific LLM support
pip install -e .[openai]     # OpenAI only
pip install -e .[anthropic]  # Anthropic only
pip install -e .[google]     # Google only
pip install -e .[all]        # All LLMs
```

### Testing
```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=dinnovos --cov-report=html

# Run specific test file
pytest tests/test_filename.py

# Run specific test function
pytest tests/test_filename.py::test_function_name
```

### Code Quality
```bash
# Format code
black dinnovos/ tests/

# Lint code
flake8 dinnovos/ tests/

# Type checking
mypy dinnovos/
```

### Building and Publishing
```bash
# Build distribution packages
python -m build

# Check distribution
twine check dist/*

# Upload to PyPI (after building)
twine upload dist/*
```

## Architecture

### Core Components

**Agent Architecture (dinnovos/core.py)**
- `Dinnovos` class: Main conversational agent that maintains conversation history and manages LLM interactions
- Maintains a message history with configurable `max_history` parameter
- Automatically keeps system prompt + last N messages to prevent context overflow
- Message format: `[{"role": "user|assistant|system", "content": "..."}]`

**LLM Abstraction Layer (dinnovos/llms/)**
- `BaseLLM`: Abstract base class defining the interface all LLM implementations must follow
- Each LLM implementation (OpenAI, Anthropic, Google) inherits from `BaseLLM`
- All implementations provide lazy imports - dependencies are only required when that specific LLM is used
- Each LLM handles provider-specific message format conversions internally

**Key Design Patterns**:
1. **Provider Abstraction**: Each LLM provider has its own message format requirements:
   - OpenAI: Standard chat completion format
   - Anthropic: Requires system message separated from conversation messages
   - Google: Uses "model" role instead of "assistant", history-based chat initialization

2. **Conversation Management**: The `Dinnovos` class implements sliding window context:
   - System prompt is always preserved at index 0
   - When history exceeds `max_history + 1`, only the most recent messages are kept
   - Implementation: `[system] + messages[-(max_history):]`

3. **Lazy Dependency Loading**: LLM implementations only import their respective SDKs in `__init__`, raising `ImportError` with helpful messages if not installed

### Module Structure

```
dinnovos/
├── __init__.py          # Public API exports
├── core.py              # Dinnovos agent implementation
├── version.py           # Version string (__version__)
└── llms/
    ├── __init__.py      # LLM exports
    ├── base.py          # BaseLLM abstract class
    ├── openai.py        # OpenAI implementation
    ├── anthropic.py     # Anthropic implementation
    └── google.py        # Google implementation
```

### Important Implementation Details

**Message History Management** (dinnovos/core.py:54-57):
The history trimming preserves conversation context while preventing memory bloat. System prompt is always kept, and the sliding window ensures recent context is available.

**Anthropic-Specific Handling** (dinnovos/llms/anthropic.py:21-32):
Anthropic's API requires the system message to be passed as a separate `system` parameter, not in the messages array. The implementation filters out system messages and passes them separately.

**Google-Specific Handling** (dinnovos/llms/google.py:22-34):
Google's Gemini uses a chat history model where all messages except the last are passed as history, and the last message is sent via `send_message()`. Role mapping: "assistant" → "model", "system" → "user".

## Configuration Files

- `setup.py`: Primary package configuration (reads version from dinnovos/version.py)
- `pyproject.toml`: Build system requirements and metadata
- Both files should be kept in sync for package information
- Version source of truth: `dinnovos/version.py`

## Common Tasks

### Adding a New LLM Provider
1. Create new file in `dinnovos/llms/` (e.g., `newprovider.py`)
2. Implement class inheriting from `BaseLLM`
3. Implement `call(messages, temperature)` method with provider-specific logic
4. Handle message format conversion if needed
5. Add lazy import with helpful `ImportError` message
6. Export in `dinnovos/llms/__init__.py`
7. Export in `dinnovos/__init__.py`
8. Add to `extras_require` in `setup.py`

### Updating Version
1. Edit `dinnovos/version.py`
2. Version is automatically read by both `setup.py` and `pyproject.toml`

### Testing LLM Implementations
When testing or debugging LLM integrations, use the examples in `examples/` as reference. Each example demonstrates the proper initialization and usage pattern.
