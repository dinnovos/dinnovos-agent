"""
Example of using AnthropicLLM with call_stream_with_function_execution method.
This demonstrates streaming responses with automatic function calling.
"""

import os
from dinnovos.llms import AnthropicLLM


# Define some example functions
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get the current weather for a location"""
    # This is a mock function - in reality, you'd call a weather API
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "condition": "sunny"
    }


def calculate(operation: str, a: float, b: float) -> dict:
    """Perform a mathematical calculation"""
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: Division by zero"
    }
    return {
        "operation": operation,
        "a": a,
        "b": b,
        "result": operations.get(operation, "Unknown operation")
    }


def search_database(query: str, limit: int = 5) -> dict:
    """Search a database with a query"""
    # Mock database search
    return {
        "query": query,
        "results": [
            {"id": 1, "title": f"Result 1 for {query}"},
            {"id": 2, "title": f"Result 2 for {query}"},
            {"id": 3, "title": f"Result 3 for {query}"}
        ][:limit]
    }


# Define tools in OpenAI format (will be converted to Anthropic format automatically)
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
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The mathematical operation to perform"
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
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search a database with a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Map function names to actual functions
available_functions = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_database": search_database
}


def main():
    # Initialize the LLM
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return
    
    llm = AnthropicLLM(api_key=api_key, model="claude-sonnet-4-5-20250929")
    
    # Example 1: Simple weather query
    print("=" * 80)
    print("Example 1: Weather Query with Streaming")
    print("=" * 80)
    
    messages = [
        {"role": "user", "content": "What's the weather like in Paris and London? Compare them."}
    ]
    
    print("\nStreaming response:")
    for chunk in llm.call_stream_with_function_execution(
        messages=messages,
        tools=tools,
        available_functions=available_functions,
        temperature=0.7,
        verbose=True,
        manage_context=True
    ):
        chunk_type = chunk.get("type")
        
        if chunk_type == "iteration_start":
            print(f"\n🔄 {chunk['content']}")
        
        elif chunk_type == "text_delta":
            # Print text chunks as they arrive
            print(chunk["content"], end="", flush=True)
        
        elif chunk_type == "function_call_start":
            print(f"\n\n🔧 Calling function: {chunk['function_name']}")
            print(f"   Arguments: {chunk['arguments']}")
        
        elif chunk_type == "function_call_result":
            print(f"   ✅ Result: {chunk['result']}")
        
        elif chunk_type == "final":
            print(f"\n\n✨ Final response: {chunk['content']}")
            print(f"   Total iterations: {chunk['iterations']}")
            print(f"   Total function calls: {len(chunk['function_calls'])}")
            if "context_stats" in chunk:
                stats = chunk["context_stats"]
                print(f"   Context usage: {stats['usage_percent']}%")
        
        elif chunk_type == "error":
            print(f"\n❌ Error: {chunk['content']}")
    
    # Example 2: Math calculation
    print("\n\n" + "=" * 80)
    print("Example 2: Math Calculation with Streaming")
    print("=" * 80)
    
    messages = [
        {"role": "user", "content": "Calculate 15 * 7, then add 23 to the result. What's the final answer?"}
    ]
    
    print("\nStreaming response:")
    for chunk in llm.call_stream_with_function_execution(
        messages=messages,
        tools=tools,
        available_functions=available_functions,
        temperature=0.7,
        verbose=False,  # Less verbose for this example
        manage_context=True
    ):
        chunk_type = chunk.get("type")
        
        if chunk_type == "text_delta":
            print(chunk["content"], end="", flush=True)
        
        elif chunk_type == "function_call_start":
            print(f"\n🔧 {chunk['function_name']}({chunk['arguments']})")
        
        elif chunk_type == "function_call_result":
            print(f"   → {chunk['result']}")
        
        elif chunk_type == "final":
            print(f"\n\n✅ Done! ({chunk['iterations']} iterations, {len(chunk['function_calls'])} function calls)")
    
    # Example 3: Database search
    print("\n\n" + "=" * 80)
    print("Example 3: Database Search with Streaming")
    print("=" * 80)
    
    messages = [
        {"role": "user", "content": "Search the database for 'machine learning' and tell me about the top 3 results."}
    ]
    
    print("\nStreaming response:")
    final_result = None
    for chunk in llm.call_stream_with_function_execution(
        messages=messages,
        tools=tools,
        available_functions=available_functions,
        temperature=0.7,
        verbose=False,
        manage_context=True
    ):
        chunk_type = chunk.get("type")
        
        if chunk_type == "text_delta":
            print(chunk["content"], end="", flush=True)
        
        elif chunk_type == "final":
            final_result = chunk
            print("\n")
    
    if final_result:
        print(f"\n📊 Summary:")
        print(f"   - Iterations: {final_result['iterations']}")
        print(f"   - Function calls: {len(final_result['function_calls'])}")
        for i, call in enumerate(final_result['function_calls'], 1):
            print(f"   - Call {i}: {call['name']}({call['arguments']})")


if __name__ == "__main__":
    main()
