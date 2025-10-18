"""Anthropic LLM interface for Dinnovos Agent"""

from typing import List, Dict, Iterator, Optional, Any, Callable
from .base import BaseLLM


class AnthropicLLM(BaseLLM):
    """Interface for Anthropic models (Claude)"""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929"):
        super().__init__(api_key, model)
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Install package: pip install anthropic")
    
    def call(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Calls Anthropic API"""
        try:
            # Anthropic requires separating the system message
            system_message = ""
            formatted_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Anthropic expects system as a list (not a string)
            # Build API call parameters
            api_params = {
                "model": self.model,
                "max_tokens": 4096,
                "temperature": temperature,
                "messages": formatted_messages
            }
            
            # Only add system parameter if there's a system message
            if system_message and system_message.strip():
                api_params["system"] = [{"type": "text", "text": system_message}]
            
            response = self.client.messages.create(**api_params)
            
            return response.content[0].text
        except Exception as e:
            return f"Error in Anthropic: {str(e)}"
    
    def stream(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Iterator[str]:
        """Streams Anthropic API response"""
        try:
            # Anthropic requires separating the system message
            system_message = ""
            formatted_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Create streaming response
            # Build API call parameters
            stream_params = {
                "model": self.model,
                "max_tokens": 4096,
                "temperature": temperature,
                "messages": formatted_messages
            }
            
            # Only add system parameter if there's a system message
            if system_message and system_message.strip():
                stream_params["system"] = [{"type": "text", "text": system_message}]
            
            with self.client.messages.stream(**stream_params) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            yield f"Error in Anthropic: {str(e)}"
    
    def call_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Calls Anthropic Claude API with function calling (tools) support.
        
        Args:
            messages: List of messages in format [{"role": "user/assistant/system", "content": "..."}]
            tools: List of tool definitions in Anthropic format
            tool_choice: "auto", "any", "none", or {"type": "tool", "name": "tool_name"}
            temperature: Temperature for generation (0-1)
        
        Returns:
            Dict with 'content' (str or None), 'tool_calls' (list or None), and 'finish_reason'
        """
        try:
            # Separate system message
            system_message = ""
            formatted_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                elif msg["role"] == "tool":
                    # Tool result message
                    formatted_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": msg["content"]
                        }]
                    })
                elif msg.get("tool_calls"):
                    # Assistant message with tool calls
                    content = []
                    if msg.get("content"):
                        content.append({"type": "text", "text": msg["content"]})
                    
                    for tool_call in msg["tool_calls"]:
                        import json
                        content.append({
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"])
                        })
                    
                    formatted_messages.append({
                        "role": "assistant",
                        "content": content
                    })
                else:
                    # Regular message
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Convert tools to Anthropic format
            anthropic_tools = self._convert_tools_to_anthropic_format(tools)
            
            # Prepare tool_choice parameter
            if tool_choice == "auto":
                tool_choice_param = {"type": "auto"}
            elif tool_choice == "any":
                tool_choice_param = {"type": "any"}
            elif tool_choice == "none":
                tool_choice_param = None
            elif isinstance(tool_choice, dict):
                tool_choice_param = tool_choice
            else:
                tool_choice_param = {"type": "auto"}
            
            # Call API
            # Build API call parameters
            api_params = {
                "model": self.model,
                "max_tokens": 4096,
                "temperature": temperature,
                "messages": formatted_messages
            }
            
            # Only add system parameter if there's a system message
            if system_message and system_message.strip():
                api_params["system"] = [{"type": "text", "text": system_message}]
            
            # Add tools if provided
            if anthropic_tools:
                api_params["tools"] = anthropic_tools
                api_params["tool_choice"] = tool_choice_param
            
            response = self.client.messages.create(**api_params)
            
            result = {
                "content": None,
                "tool_calls": None,
                "finish_reason": response.stop_reason
            }
            
            # Process response content
            text_content = []
            tool_calls = []
            
            for block in response.content:
                if block.type == "text":
                    text_content.append(block.text)
                elif block.type == "tool_use":
                    import json
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input)
                        }
                    })
            
            if text_content:
                result["content"] = "".join(text_content)
            
            if tool_calls:
                result["tool_calls"] = tool_calls
                result["finish_reason"] = "tool_calls"
            
            return result
        except Exception as e:
            return {
                "content": f"Error in Anthropic: {str(e)}",
                "tool_calls": None,
                "finish_reason": "error"
            }
    
    def stream_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = 0.7
    ) -> Iterator[Dict[str, Any]]:
        """
        Streams Anthropic Claude API response with function calling (tools) support.
        
        Args:
            messages: List of messages
            tools: List of tool definitions in Anthropic format
            tool_choice: "auto", "any", "none", or specific tool
            temperature: Temperature for generation (0-1)
        
        Yields:
            Dict chunks with 'type' ('content' or 'tool_call'), 'delta' (content chunk),
            'tool_call_id', 'function_name', 'function_arguments'
        """
        try:
            # Separate system message
            system_message = ""
            formatted_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                elif msg["role"] == "tool":
                    formatted_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": msg["content"]
                        }]
                    })
                elif msg.get("tool_calls"):
                    content = []
                    if msg.get("content"):
                        content.append({"type": "text", "text": msg["content"]})
                    
                    for tool_call in msg["tool_calls"]:
                        import json
                        content.append({
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"])
                        })
                    
                    formatted_messages.append({
                        "role": "assistant",
                        "content": content
                    })
                else:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Convert tools
            anthropic_tools = self._convert_tools_to_anthropic_format(tools)
            
            # Prepare tool_choice
            if tool_choice == "auto":
                tool_choice_param = {"type": "auto"}
            elif tool_choice == "any":
                tool_choice_param = {"type": "any"}
            elif tool_choice == "none":
                tool_choice_param = None
            elif isinstance(tool_choice, dict):
                tool_choice_param = tool_choice
            else:
                tool_choice_param = {"type": "auto"}
            
            # Stream response
            # Build API call parameters
            stream_params = {
                "model": self.model,
                "max_tokens": 4096,
                "temperature": temperature,
                "messages": formatted_messages
            }
            
            # Only add system parameter if there's a system message
            if system_message and system_message.strip():
                stream_params["system"] = [{"type": "text", "text": system_message}]
            
            # Add tools if provided
            if anthropic_tools:
                stream_params["tools"] = anthropic_tools
                stream_params["tool_choice"] = tool_choice_param
            
            with self.client.messages.stream(**stream_params) as stream:
                for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, 'text'):
                                yield {
                                    "type": "content",
                                    "delta": event.delta.text,
                                    "finish_reason": None
                                }
                            elif hasattr(event.delta, 'partial_json'):
                                # Tool use in progress
                                yield {
                                    "type": "tool_call",
                                    "index": event.index,
                                    "tool_call_id": None,
                                    "function_name": None,
                                    "function_arguments": event.delta.partial_json,
                                    "finish_reason": None
                                }
                        elif event.type == "content_block_start":
                            if hasattr(event.content_block, 'type') and event.content_block.type == "tool_use":
                                yield {
                                    "type": "tool_call",
                                    "index": event.index,
                                    "tool_call_id": event.content_block.id,
                                    "function_name": event.content_block.name,
                                    "function_arguments": "",
                                    "finish_reason": None
                                }
        except Exception as e:
            yield {
                "type": "error",
                "delta": f"Error in Anthropic: {str(e)}",
                "finish_reason": "error"
            }
    
    def call_with_function_execution(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        available_functions: Dict[str, Callable],
        tool_choice: str = "auto",
        temperature: float = 0.7,
        max_iterations: int = 5,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Flexible method that automatically handles the complete function calling cycle:
        1. Calls the LLM with tools
        2. Executes the requested functions
        3. Sends the results back to the LLM
        4. Repeats until getting a final response or reaching max_iterations
        
        Args:
            messages: Initial list of messages
            tools: Tool definitions in OpenAI/Anthropic format
            available_functions: Dict mapping function names to callables
            tool_choice: "auto", "any", "none", or specific tool
            temperature: Temperature for generation
            max_iterations: Maximum number of iterations to prevent infinite loops
            verbose: If True, prints debug information
        
        Returns:
            Dict with:
                - 'content': Final LLM response
                - 'messages': Complete message history
                - 'function_calls': List of all functions called
                - 'iterations': Number of iterations performed
        """
        import json
        
        conversation_messages = messages.copy()
        all_function_calls = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration}")
                print(f"{'='*60}")
            
            # Call the LLM with tools
            response = self.call_with_tools(
                messages=conversation_messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature
            )
            
            # If there's content and no tool calls, we're done
            if response["content"] and not response["tool_calls"]:
                if verbose:
                    print(f"\n✅ Final response: {response['content']}")
                
                return {
                    "content": response["content"],
                    "messages": conversation_messages,
                    "function_calls": all_function_calls,
                    "iterations": iteration,
                    "finish_reason": response["finish_reason"]
                }
            
            # If there are no tool calls, something went wrong
            if not response["tool_calls"]:
                if verbose:
                    print("⚠️ No tool calls or content")
                
                return {
                    "content": response.get("content") or "No response generated",
                    "messages": conversation_messages,
                    "function_calls": all_function_calls,
                    "iterations": iteration,
                    "finish_reason": response["finish_reason"]
                }
            
            # Add assistant message with tool calls
            conversation_messages.append({
                "role": "assistant",
                "content": response.get("content"),
                "tool_calls": response["tool_calls"]
            })
            
            # Execute each tool call
            for tool_call in response["tool_calls"]:
                function_name = tool_call["function"]["name"]
                function_args_str = tool_call["function"]["arguments"]
                
                try:
                    function_args = json.loads(function_args_str)
                except json.JSONDecodeError as e:
                    function_args = {}
                    if verbose:
                        print(f"⚠️ Error parsing arguments: {e}")
                
                if verbose:
                    print(f"\n🔧 Calling function: {function_name}")
                    print(f"📋 Arguments: {function_args}")
                
                # Verify that the function exists
                if function_name not in available_functions:
                    error_msg = f"Function '{function_name}' not found in available_functions"
                    if verbose:
                        print(f"❌ {error_msg}")
                    
                    function_response = json.dumps({"error": error_msg})
                else:
                    # Execute the function
                    try:
                        function_to_call = available_functions[function_name]
                        result = function_to_call(**function_args)
                        
                        # Ensure the result is a string
                        if isinstance(result, str):
                            function_response = result
                        else:
                            function_response = json.dumps(result)
                        
                        if verbose:
                            print(f"✅ Result: {function_response}")
                        
                    except Exception as e:
                        error_msg = f"Error executing function: {str(e)}"
                        if verbose:
                            print(f"❌ {error_msg}")
                        function_response = json.dumps({"error": error_msg})
                
                # Register the call
                all_function_calls.append({
                    "name": function_name,
                    "arguments": function_args,
                    "result": function_response
                })
                
                # Add the function result to messages
                conversation_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": function_name,
                    "content": function_response
                })
        
        # If we get here, we reached max_iterations
        if verbose:
            print(f"\n⚠️ Maximum iterations reached ({max_iterations})")
        
        return {
            "content": "Maximum iterations reached without final response",
            "messages": conversation_messages,
            "function_calls": all_function_calls,
            "iterations": iteration,
            "finish_reason": "max_iterations"
        }
    
    def _convert_tools_to_anthropic_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI format tools to Anthropic format"""
        anthropic_tools = []
        
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                
                # Anthropic uses a similar format to OpenAI
                anthropic_tool = {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})
                }
                anthropic_tools.append(anthropic_tool)
        
        return anthropic_tools