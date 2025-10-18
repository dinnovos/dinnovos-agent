# Function Calling (Tools) in Dinnovos Agent

This guide explains how to use function calling (also known as "tools") with OpenAI in Dinnovos Agent.

## 🚀 Recommended Methods

### `call_with_function_execution()` - Standard Execution

**The easiest way** to use function calling is with the `call_with_function_execution()` method, which automatically handles the entire cycle:

```python
import os
from dinnovos.llms.openai import OpenAILLM

# 1. Initialize
llm = OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

# 2. Define functions
def get_weather(location: str) -> dict:
    return {"temp": 22, "condition": "sunny"}

available_functions = {"get_weather": get_weather}

# 3. Define tools
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

# 4. One line does it all!
result = llm.call_with_function_execution(
    messages=[{"role": "user", "content": "What's the weather in Bogotá?"}],
    tools=tools,
    available_functions=available_functions,
    verbose=True  # See the process
)

print(result["content"])  # Final response
```

### Advantages

✅ **Simple**: One call instead of multiple steps  
✅ **Automatic**: Executes functions and handles the complete cycle  
✅ **Safe**: Prevents infinite loops with `max_iterations`  
✅ **Debugging**: `verbose` mode to see the entire process  
✅ **Flexible**: Supports multiple iterations and chained functions  

### Parameters

- **`messages`**: Initial list of messages
- **`tools`**: Tool definitions in OpenAI format
- **`available_functions`**: Dict with executable functions
- **`tool_choice`**: `"auto"` (default), `"none"`, or specific
- **`temperature`**: Generation temperature (default: 0.7)
- **`max_iterations`**: Maximum iterations (default: 5)
- **`verbose`**: Show debug information (default: False)

### Returns

```python
{
    "content": "Final LLM response",
    "messages": [...],  # Complete history
    "function_calls": [...],  # All functions called
    "iterations": 2,  # Number of iterations
    "finish_reason": "stop"  # Finish reason
}
```

---

### `call_stream_with_function_execution()` - Streaming Execution

**For real-time responses**, use the `call_stream_with_function_execution()` method, which streams responses while automatically executing functions:

```python
import os
from dinnovos.llms.openai import OpenAILLM

# 1. Initialize
llm = OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

# 2. Define functions
def get_weather(location: str) -> dict:
    return {"temp": 22, "condition": "sunny"}

available_functions = {"get_weather": get_weather}

# 3. Define tools (same format as above)
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

# 4. Stream with automatic function execution
for chunk in llm.call_stream_with_function_execution(
    messages=[{"role": "user", "content": "What's the weather in Bogotá?"}],
    tools=tools,
    available_functions=available_functions,
    verbose=True
):
    chunk_type = chunk.get("type")
    
    if chunk_type == "text_delta":
        # Stream text as it arrives
        print(chunk.get("content"), end="", flush=True)
    
    elif chunk_type == "function_call_start":
        # Function is being called
        print(f"\n🔧 Calling: {chunk.get('function_name')}")
    
    elif chunk_type == "function_call_result":
        # Function completed
        print(f"✅ Result: {chunk.get('result')}")
    
    elif chunk_type == "final":
        # Final response
        print(f"\n\nCompleted in {chunk.get('iterations')} iterations")
```

### Advantages of Streaming

✅ **Real-time**: See responses as they're generated  
✅ **Better UX**: Users see progress immediately  
✅ **Transparent**: See function calls as they happen  
✅ **Automatic**: Same automatic execution as standard method  
✅ **Flexible**: Handle different event types as needed  

### Stream Event Types

The streaming method yields dictionaries with different types:

- **`iteration_start`**: New iteration begins
  - `iteration`: Current iteration number
  - `content`: Iteration message

- **`text_delta`**: Text chunk from LLM
  - `content`: Text chunk to display
  - `iteration`: Current iteration

- **`function_call_start`**: Function is about to be called
  - `function_name`: Name of the function
  - `arguments`: Parsed arguments dict
  - `iteration`: Current iteration

- **`function_call_result`**: Function execution completed
  - `function_name`: Name of the function
  - `result`: Function result (string)
  - `iteration`: Current iteration

- **`final`**: Final response received
  - `content`: Final text response
  - `messages`: Complete message history
  - `function_calls`: All functions called
  - `iterations`: Total iterations
  - `context_stats`: Context usage (if manage_context=True)

- **`error`**: Error occurred
  - `content`: Error message
  - `iteration`: Current iteration

### Parameters (same as standard method)

- **`messages`**: Initial list of messages
- **`tools`**: Tool definitions in OpenAI format
- **`available_functions`**: Dict with executable functions
- **`tool_choice`**: `"auto"` (default), `"none"`, or specific
- **`temperature`**: Generation temperature (default: 0.7)
- **`max_iterations`**: Maximum iterations (default: 5)
- **`verbose`**: Show debug information (default: False)
- **`manage_context`**: Enable context management (default: True)

---

## What is Function Calling?

Function calling allows language models to call external functions in a structured way. The model can:
1. Decide when it needs to call a function
2. Generate the correct arguments in JSON format
3. Receive the result and use it to generate a final response

## Available Methods

### `call_with_tools()`

Performs a complete call with tools support.

```python
response = llm.call_with_tools(
    messages=messages,
    tools=tools,
    tool_choice="auto",  # "auto", "none", or specific
    temperature=0.7
)
```

**Returns:**
```python
{
    "content": "response text or None",
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "function_name",
                "arguments": '{"param": "value"}'
            }
        }
    ],
    "finish_reason": "tool_calls" or "stop"
}
```

### `stream_with_tools()`

Performs streaming with tools support.

```python
for chunk in llm.stream_with_tools(messages, tools):
    if chunk["type"] == "content":
        print(chunk["delta"], end="")
    elif chunk["type"] == "tool_call":
        # Process tool call
        pass
```

## Tools Format

Tools are defined in OpenAI format:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "function_name",
            "description": "Clear description of what the function does",
            "parameters": {
                "type": "object",
                "properties": {
                    "parameter1": {
                        "type": "string",
                        "description": "Parameter description"
                    },
                    "parameter2": {
                        "type": "number",
                        "description": "Another parameter"
                    }
                },
                "required": ["parameter1"]
            }
        }
    }
]
```

## Complete Example

```python
import os
import json
from dinnovos.llms.openai import OpenAILLM

# 1. Initialize the LLM
llm = OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

# 2. Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# 3. Implement the function
def get_weather(location: str) -> str:
    # Your logic here
    return json.dumps({"temp": 22, "condition": "sunny"})

available_functions = {"get_weather": get_weather}

# 4. Make the call
messages = [{"role": "user", "content": "What's the weather in Bogotá?"}]
response = llm.call_with_tools(messages, tools)

# 5. Execute tool calls if they exist
if response["tool_calls"]:
    for tool_call in response["tool_calls"]:
        function_name = tool_call["function"]["name"]
        function_args = json.loads(tool_call["function"]["arguments"])
        
        # Execute the function
        function_response = available_functions[function_name](**function_args)
        
        # Add to messages
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
    
    # 6. Get final response
    final_response = llm.call_with_tools(messages, tools)
    print(final_response["content"])
```

## `tool_choice` Parameter

- `"auto"` (default): The model decides whether to call functions
- `"none"`: The model will not call functions
- `{"type": "function", "function": {"name": "function_name"}}`: Forces a specific function

## Streaming with Tools

```python
messages = [{"role": "user", "content": "Calculate 2 + 2"}]
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

# Execute accumulated functions
for idx, tool_call_data in tool_calls_buffer.items():
    function_name = tool_call_data["name"]
    function_args = json.loads(tool_call_data["arguments"])
    result = available_functions[function_name](**function_args)
    print(f"\nResult: {result}")
```

## Best Practices

1. **Clear descriptions**: Write detailed descriptions of functions and parameters
2. **Validation**: Validate arguments before executing functions
3. **Error handling**: Catch exceptions and return errors in JSON format
4. **Security**: Never execute arbitrary code with `eval()` in production
5. **Data types**: Use the correct types in the schema (string, number, boolean, array, object)

## Included Examples

- `openai_function_calling.py`: Complete example with multiple functions
- `openai_stream_function_calling.py`: Streaming example with automatic function execution
- `openai_tools.ipynb`: Interactive notebook with step-by-step examples

## Supported Parameter Types

```python
"parameters": {
    "type": "object",
    "properties": {
        "string_param": {"type": "string"},
        "number_param": {"type": "number"},
        "integer_param": {"type": "integer"},
        "boolean_param": {"type": "boolean"},
        "array_param": {
            "type": "array",
            "items": {"type": "string"}
        },
        "enum_param": {
            "type": "string",
            "enum": ["option1", "option2"]
        },
        "object_param": {
            "type": "object",
            "properties": {
                "nested": {"type": "string"}
            }
        }
    },
    "required": ["string_param"]
}
```

## Troubleshooting

### The model doesn't call functions
- Verify that the description is clear
- Make sure `tool_choice` is not set to `"none"`
- Try with a more explicit prompt

### Error parsing arguments
- Verify that the parameter schema is valid
- Check that data types are correct
- Use `json.loads()` to parse arguments

### Functions not available
- Make sure all functions are in `available_functions`
- Verify that names match exactly
