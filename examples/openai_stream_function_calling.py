"""Example: OpenAI streaming with automatic function execution"""

import os
from dinnovos.llms import OpenAILLM

# Example functions that the LLM can call
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location"""
    # Simulated weather data
    weather_data = {
        "New York": {"celsius": "15°C", "fahrenheit": "59°F"},
        "London": {"celsius": "12°C", "fahrenheit": "54°F"},
        "Tokyo": {"celsius": "18°C", "fahrenheit": "64°F"},
    }
    
    temp = weather_data.get(location, {}).get(unit, "Unknown")
    return f"The weather in {location} is {temp}"

def calculate(operation: str, a: float, b: float) -> float:
    """Perform a mathematical operation"""
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: Division by zero"
    }
    return operations.get(operation, "Unknown operation")

# Tool definitions in OpenAI format
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
                        "description": "The city name, e.g., New York, London"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical operation",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The operation to perform"
                    },
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["operation", "a", "b"]
            }
        }
    }
]

# Available functions mapping
available_functions = {
    "get_weather": get_weather,
    "calculate": calculate
}

def main():
    # Initialize OpenAI LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    llm = OpenAILLM(api_key=api_key, model="gpt-4")
    
    # Initial messages
    messages = [
        {
            "role": "user",
            "content": "What's the weather in New York? Also, what's 15 multiplied by 7?"
        }
    ]
    
    print("=" * 60)
    print("STREAMING WITH FUNCTION EXECUTION")
    print("=" * 60)
    print(f"\nUser: {messages[0]['content']}\n")
    print("Assistant: ", end="", flush=True)
    
    # Stream with automatic function execution
    final_result = None
    for chunk in llm.call_stream_with_function_execution(
        messages=messages,
        tools=tools,
        available_functions=available_functions,
        temperature=0.7,
        max_iterations=5,
        verbose=False,  # Set to True to see debug info
        manage_context=True
    ):
        chunk_type = chunk.get("type")
        
        if chunk_type == "iteration_start":
            # New iteration started
            iteration = chunk.get("iteration")
            if iteration > 1:
                print(f"\n\n[Iteration {iteration}]")
                print("Assistant: ", end="", flush=True)
        
        elif chunk_type == "text_delta":
            # Stream text as it arrives
            print(chunk.get("content", ""), end="", flush=True)
        
        elif chunk_type == "function_call_start":
            # Function is about to be called
            func_name = chunk.get("function_name")
            args = chunk.get("arguments")
            print(f"\n\n🔧 Calling function: {func_name}")
            print(f"📋 Arguments: {args}")
        
        elif chunk_type == "function_call_result":
            # Function execution completed
            func_name = chunk.get("function_name")
            result = chunk.get("result")
            print(f"✅ Result from {func_name}: {result}")
        
        elif chunk_type == "final":
            # Final response received
            final_result = chunk
            print("\n")
        
        elif chunk_type == "error":
            # Error occurred
            print(f"\n❌ Error: {chunk.get('content')}")
            break
    
    # Print summary
    if final_result:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total iterations: {final_result.get('iterations')}")
        print(f"Functions called: {len(final_result.get('function_calls', []))}")
        
        for i, func_call in enumerate(final_result.get('function_calls', []), 1):
            print(f"\n{i}. {func_call['name']}")
            print(f"   Args: {func_call['arguments']}")
            print(f"   Result: {func_call['result']}")
        
        if final_result.get('context_stats'):
            stats = final_result['context_stats']
            print(f"\nContext usage: {stats['current_tokens']} / {stats['max_tokens']} tokens ({stats['usage_percent']}%)")

if __name__ == "__main__":
    main()
