"""Example of using OpenAI with function calling (tools)"""

import os
import json
from dinnovos.llms.openai import OpenAILLM


# Define tools/functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
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
                        "description": "The temperature unit to use"
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
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g. '2 + 2'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


# Mock function implementations
def get_weather(location: str, unit: str = "celsius") -> str:
    """Mock weather function"""
    return json.dumps({
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "Sunny"
    })


def calculate(expression: str) -> str:
    """Mock calculator function"""
    try:
        result = eval(expression)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


# Function dispatcher
available_functions = {
    "get_weather": get_weather,
    "calculate": calculate
}


def main():
    # Initialize OpenAI LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    llm = OpenAILLM(api_key=api_key, model="gpt-4")
    
    # Example 1: Simple function call
    print("=" * 60)
    print("Example 1: Simple function call")
    print("=" * 60)
    
    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ]
    
    response = llm.call_with_tools(messages, tools)
    print(f"\nResponse: {response}")
    
    # If there are tool calls, execute them
    if response["tool_calls"]:
        for tool_call in response["tool_calls"]:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            
            print(f"\nCalling function: {function_name}")
            print(f"Arguments: {function_args}")
            
            # Execute the function
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)
            
            print(f"Function response: {function_response}")
            
            # Add the function response to messages
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": response["tool_calls"]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": function_name,
                "content": function_response
            })
        
        # Get final response
        final_response = llm.call_with_tools(messages, tools)
        print(f"\nFinal response: {final_response['content']}")
    
    # Example 2: Streaming with tools
    print("\n" + "=" * 60)
    print("Example 2: Streaming with function calls")
    print("=" * 60)
    
    messages = [
        {"role": "user", "content": "Calculate 25 * 4 + 10"}
    ]
    
    print("\nStreaming response:")
    tool_calls_buffer = {}
    
    for chunk in llm.stream_with_tools(messages, tools):
        if chunk["type"] == "content":
            print(chunk["delta"], end="", flush=True)
        elif chunk["type"] == "tool_call":
            idx = chunk["index"]
            if idx not in tool_calls_buffer:
                tool_calls_buffer[idx] = {
                    "id": "",
                    "name": "",
                    "arguments": ""
                }
            
            if chunk["tool_call_id"]:
                tool_calls_buffer[idx]["id"] = chunk["tool_call_id"]
            if chunk["function_name"]:
                tool_calls_buffer[idx]["name"] = chunk["function_name"]
            if chunk["function_arguments"]:
                tool_calls_buffer[idx]["arguments"] += chunk["function_arguments"]
    
    print()
    
    # Execute tool calls from streaming
    if tool_calls_buffer:
        for idx, tool_call_data in tool_calls_buffer.items():
            function_name = tool_call_data["name"]
            function_args = json.loads(tool_call_data["arguments"])
            
            print(f"\nCalling function: {function_name}")
            print(f"Arguments: {function_args}")
            
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)
            
            print(f"Function response: {function_response}")


if __name__ == "__main__":
    main()
