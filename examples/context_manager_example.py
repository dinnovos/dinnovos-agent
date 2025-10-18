"""Example of using ContextManager with AnthropicLLM"""

from dinnovos.llms.anthropic import AnthropicLLM
import os

def main():
    # Initialize AnthropicLLM with context management
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Create LLM with custom context settings
    llm = AnthropicLLM(
        api_key=api_key,
        model="claude-sonnet-4-5-20250929",
        max_tokens=100000,  # Maximum context window
        context_strategy="smart"  # Options: "fifo", "smart", "summary"
    )
    
    # Simulate a long conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language..."},
        {"role": "user", "content": "Tell me about its history."},
        {"role": "assistant", "content": "Python was created by Guido van Rossum..."},
    ]
    
    # Add many more messages to exceed context limit
    for i in range(50):
        messages.append({
            "role": "user",
            "content": f"This is message number {i} with some content to fill the context window."
        })
        messages.append({
            "role": "assistant",
            "content": f"Response to message {i} with additional content."
        })
    
    # Get context statistics before calling
    print("=" * 60)
    print("CONTEXT STATISTICS BEFORE CALL")
    print("=" * 60)
    stats = llm.get_context_stats(messages)
    print(f"Current tokens: {stats['current_tokens']}")
    print(f"Max tokens: {stats['max_tokens']}")
    print(f"Available tokens: {stats['available_tokens']}")
    print(f"Usage: {stats['usage_percent']}%")
    print(f"Messages count: {stats['messages_count']}")
    print(f"Within limit: {stats['within_limit']}")
    print()
    
    # Make a call with context management enabled (default)
    print("=" * 60)
    print("CALLING LLM WITH CONTEXT MANAGEMENT")
    print("=" * 60)
    
    messages.append({
        "role": "user",
        "content": "Can you summarize our conversation?"
    })
    
    response = llm.call(
        messages=messages,
        temperature=0.7,
        manage_context=True,  # Enable context management
        verbose=True  # Show context management details
    )
    
    print("\n" + "=" * 60)
    print("RESPONSE")
    print("=" * 60)
    print(response)
    print()
    
    # Get statistics after call
    print("=" * 60)
    print("CONTEXT STATISTICS AFTER CALL")
    print("=" * 60)
    stats = llm.get_context_stats(messages)
    print(f"Truncated count: {stats['truncated_count']}")
    print(f"Total tokens saved: {stats['total_tokens_saved']}")
    print()
    
    # Example with streaming
    print("=" * 60)
    print("STREAMING WITH CONTEXT MANAGEMENT")
    print("=" * 60)
    
    messages.append({
        "role": "user",
        "content": "Tell me a short story."
    })
    
    print("Streaming response: ", end="", flush=True)
    for chunk in llm.stream(messages=messages, manage_context=True, verbose=False):
        print(chunk, end="", flush=True)
    print("\n")
    
    # Example with function calling
    print("=" * 60)
    print("FUNCTION CALLING WITH CONTEXT MANAGEMENT")
    print("=" * 60)
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    def get_weather(location: str, unit: str = "celsius") -> str:
        """Mock weather function"""
        return f"The weather in {location} is 22°{unit[0].upper()}"
    
    available_functions = {
        "get_weather": get_weather
    }
    
    weather_messages = [
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "What's the weather in Paris?"}
    ]
    
    result = llm.call_with_function_execution(
        messages=weather_messages,
        tools=tools,
        available_functions=available_functions,
        manage_context=True,  # Context management also works with function calling
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("FUNCTION CALLING RESULT")
    print("=" * 60)
    print(f"Final response: {result['content']}")
    print(f"Functions called: {len(result['function_calls'])}")
    print(f"Iterations: {result['iterations']}")
    print()
    
    # Reset statistics
    llm.reset_context_stats()
    print("Context statistics reset.")

if __name__ == "__main__":
    main()
