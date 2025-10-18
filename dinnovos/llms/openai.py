"""OpenAI LLM interface for Dinnovos Agent"""

from typing import List, Dict, Iterator, Optional, Any, Callable
from .base import BaseLLM


class OpenAILLM(BaseLLM):
    """Interface for OpenAI models (GPT-4, GPT-3.5, etc.)"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__(api_key, model)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Install package: pip install openai")
    
    def call(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Calls OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in OpenAI: {str(e)}"
    
    def stream(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Iterator[str]:
        """Streams OpenAI API response"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error in OpenAI: {str(e)}"
    
    def call_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Calls OpenAI API with function calling (tools) support.
        
        Args:
            messages: List of messages in format [{"role": "user/assistant/system", "content": "..."}]
            tools: List of tool definitions in OpenAI format
            tool_choice: "auto", "none", or {"type": "function", "function": {"name": "function_name"}}
            temperature: Temperature for generation (0-1)
        
        Returns:
            Dict with 'content' (str or None), 'tool_calls' (list or None), and 'finish_reason'
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature
            )
            
            message = response.choices[0].message
            
            result = {
                "content": message.content,
                "tool_calls": None,
                "finish_reason": response.choices[0].finish_reason
            }
            
            # Check if there are tool calls
            if message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in message.tool_calls
                ]
            
            return result
        except Exception as e:
            return {
                "content": f"Error in OpenAI: {str(e)}",
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
        Streams OpenAI API response with function calling (tools) support.
        
        Args:
            messages: List of messages
            tools: List of tool definitions in OpenAI format
            tool_choice: "auto", "none", or specific tool
            temperature: Temperature for generation (0-1)
        
        Yields:
            Dict chunks with 'type' ('content' or 'tool_call'), 'delta' (content chunk),
            'tool_call_id', 'function_name', 'function_arguments'
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                stream=True
            )
            
            for chunk in stream:
                delta = chunk.choices[0].delta
                
                # Content chunk
                if delta.content is not None:
                    yield {
                        "type": "content",
                        "delta": delta.content,
                        "finish_reason": chunk.choices[0].finish_reason
                    }
                
                # Tool call chunks
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        yield {
                            "type": "tool_call",
                            "index": tool_call.index,
                            "tool_call_id": tool_call.id if tool_call.id else None,
                            "function_name": tool_call.function.name if tool_call.function.name else None,
                            "function_arguments": tool_call.function.arguments if tool_call.function.arguments else "",
                            "finish_reason": chunk.choices[0].finish_reason
                        }
        except Exception as e:
            yield {
                "type": "error",
                "delta": f"Error in OpenAI: {str(e)}",
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
            tools: Tool definitions in OpenAI format
            available_functions: Dict mapping function names to callables
            tool_choice: "auto", "none", or specific tool
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