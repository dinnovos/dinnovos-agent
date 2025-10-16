"""
Multi-LLM example for Dinnovos Agent
Demonstrates switching between different LLM providers
"""

from dinnovos import Agent, OpenAILLM, AnthropicLLM, GoogleLLM

# Same question for all LLMs
question = "Explain what a neural network is in one sentence."

print("Dinnovos Agent - Multi-LLM Comparison")
print("=" * 60)
print(f"\nQuestion: {question}\n")

# OpenAI GPT
print("-" * 60)
print("Using OpenAI GPT-4:")
try:
    openai_llm = OpenAILLM(api_key="your-openai-key", model="gpt-4")
    agent_gpt = Agent(llm=openai_llm)
    response_gpt = agent_gpt.chat(question)
    print(f"Response: {response_gpt}")
except Exception as e:
    print(f"Error: {e}")

# Anthropic Claude
print("\n" + "-" * 60)
print("Using Anthropic Claude:")
try:
    anthropic_llm = AnthropicLLM(
        api_key="your-anthropic-key",
        model="claude-sonnet-4-5-20250929"
    )
    agent_claude = Agent(llm=anthropic_llm)
    response_claude = agent_claude.chat(question)
    print(f"Response: {response_claude}")
except Exception as e:
    print(f"Error: {e}")

# Google Gemini
print("\n" + "-" * 60)
print("Using Google Gemini:")
try:
    google_llm = GoogleLLM(api_key="your-google-key", model="gemini-1.5-pro")
    agent_gemini = Agent(llm=google_llm)
    response_gemini = agent_gemini.chat(question)
    print(f"Response: {response_gemini}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)


# Example: Switching LLMs dynamically
def create_agent_with_provider(provider: str):
    """Factory function to create Dinnovos with different providers"""

    llm_configs = {
        "openai": lambda: OpenAILLM(api_key="your-openai-key", model="gpt-4"),
        "anthropic": lambda: AnthropicLLM(api_key="your-anthropic-key"),
        "google": lambda: GoogleLLM(api_key="your-google-key")
    }

    if provider not in llm_configs:
        raise ValueError(f"Unknown provider: {provider}")

    llm = llm_configs[provider]()
    return Agent(llm=llm)


# Usage
print("\nDynamic LLM Selection:")
provider = "openai"  # Change this to switch providers
dinnovos = create_agent_with_provider(provider)
print(f"Selected provider: {provider}")