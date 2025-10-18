"""OpenAI LLM interface for Dinnovos Agent"""

from typing import List, Dict, Iterator, Optional, Any, Callable
from .base import BaseLLM
from ..utils import ContextManager


class OpenAILLM(BaseLLM):
    """Interface for OpenAI models (GPT-4, GPT-3.5, etc.)"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        
        super().__init__(api_key, model)

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Install package: pip install openai")
        
        # Initialize context manager with model-specific limits
        model_limits = {
            "gpt-4": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
        }

        max_tokens = model_limits.get(model, 128000)  # Default to GPT-4 limit
        
        self.context_manager = ContextManager(
            max_tokens=max_tokens,
            strategy="smart",
            reserve_tokens=4096
        )
    
    def call(self, messages: List[Dict[str, str]], temperature: float = 0.7, manage_context: bool = True, format: str = 'text') -> str:
        """Calls OpenAI API with optional context management using Responses API
        
        Args:
            messages: List of messages in format [{"role": "user/assistant/system", "content": "..."}]
            temperature: Temperature for generation (0-1)
            manage_context: If True, applies context management to messages before sending
            format: Response format - 'text', 'json_object', or 'json_schema' (default: 'text')
        
        Returns:
            str: The LLM response content
        """
        try:
            # Manage context if enabled
            if manage_context:
                messages = self.context_manager.manage(messages)
            
            inputs = self._normalize_messages_to_inputs(messages)

            params = {
                "model": self.model,
                "input": inputs,
                "temperature": temperature
            }
            
            # Add response_format based on format parameter
            if format == 'json_object':
                params["response_format"] = {"type": "json_object"}
            elif format == 'json_schema':
                params["response_format"] = {"type": "json_schema"}
                
            # 'text' is the default, no need to specify
            
            # Use new Responses API
            response = self.client.responses.create(**params)
            
            # Extract content from new response structure
            if getattr(response, "output_text", None):
                return response.output_text
                
            if getattr(response, "output", None):
                texts = []
                for item in response.output:
                    content_blocks = getattr(item, "content", None)
                    if content_blocks:
                        for block in content_blocks:
                            text_value = getattr(block, "text", None)
                            if text_value:
                                texts.append(text_value)
                if texts:
                    return "".join(texts)
            
            return ""
        except Exception as e:
            return f"Error in OpenAI: {str(e)}"
    
    def call_stream(self, messages: List[Dict[str, str]], temperature: float = 0.7, manage_context: bool = True, format: str = 'text') -> Iterator[str]:
        """Streams OpenAI API response with optional context management
        
        Args:
            messages: List of messages in format [{"role": "user/assistant/system", "content": "..."}]
            temperature: Temperature for generation (0-1)
            manage_context: If True, applies context management to messages before sending
            format: Response format - 'text', 'json_object', or 'json_schema' (default: 'text')
        
        Yields:
            str: Content chunks from the LLM response
        """
        try:
            # Manage context if enabled
            if manage_context:
                messages = self.context_manager.manage(messages)

            inputs = self._normalize_messages_to_inputs(messages)

            params = {
                "model": self.model,
                "input": inputs,
                "temperature": temperature
            }

            # Add response_format based on format parameter
            if format == 'json_object':
                params["response_format"] = {"type": "json_object"}
            elif format == 'json_schema':
                params["response_format"] = {"type": "json_schema"}
            # 'text' is the default, no need to specify

            has_yielded = False
            final_response = None

            try:
                with self.client.responses.stream(**params) as stream:
                    for event in stream:
                        
                        event_type = getattr(event, "type", "")

                        if event_type == "response.output_text.delta":
                            
                            delta_text = getattr(event, "delta", None)

                            if delta_text:
                                has_yielded = True
                                yield delta_text

                        elif event_type in {"response.error", "response.failed"}:
                            error = getattr(event, "error", None)
                            message = getattr(error, "message", None) if error else None
                            if message:
                                yield f"Error in OpenAI: {message}"

                    final_response = stream.get_final_response()

            except Exception as stream_error:
                yield f"Error in OpenAI: {str(stream_error)}"
                return

            if not has_yielded and final_response is not None:
                if getattr(final_response, "output_text", None):
                    yield final_response.output_text

                elif getattr(final_response, "output", None):
                    texts = []
                    
                    for item in final_response.output:
                        content_blocks = getattr(item, "content", None)
                        if content_blocks:
                            for block in content_blocks:
                                text_value = getattr(block, "text", None)
                                if text_value:
                                    texts.append(text_value)
                    if texts:
                        yield "".join(texts)

        except Exception as e:
            yield f"Error in OpenAI: {str(e)}"
    
    def call_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = 0.7,
        manage_context: bool = False,
        format: str = 'text'
    ) -> Dict[str, Any]:
        """
        Calls OpenAI API with function calling (tools) support using Responses API.
        
        Args:
            messages: List of messages in format [{"role": "user/assistant/system", "content": "..."}]
            tools: List of tool definitions in OpenAI format
            tool_choice: "auto", "none", or {"type": "function", "name": "function_name"}
            temperature: Temperature for generation (0-1)
            manage_context: If True, applies context management to messages before sending (default: False)
            format: Response format - 'text', 'json_object', or 'json_schema' (default: 'text')
        
        Returns:
            Dict with 'content' (str or None), 'tool_calls' (list or None), and 'finish_reason'
        """
        try:
            # Manage context if enabled
            if manage_context:
                messages = self.context_manager.manage(messages)

            # Convert messages to input format for Responses API
            inputs = self._normalize_messages_to_inputs(messages)

            # Normalize tools to Responses API format
            normalized_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func_def = tool.get("function", {})
                    normalized_tools.append({
                        "type": "function",
                        "name": func_def.get("name"),
                        "description": func_def.get("description"),
                        "parameters": func_def.get("parameters"),
                    })

            params = {
                "model": self.model,
                "input": inputs,
                "tools": normalized_tools,
                "tool_choice": tool_choice,
                "temperature": temperature
            }

            if format == 'json_object':
                params["response_format"] = {"type": "json_object"}
            elif format == 'json_schema':
                params["response_format"] = {"type": "json_schema"}

            response = self.client.responses.create(**params)

            # Parse response according to Responses API format
            content_segments = []
            tool_calls: List[Dict[str, Any]] = []

            # Extract output_text if available
            if hasattr(response, "output_text") and response.output_text:
                content_segments.append(response.output_text)

            # Parse output array for function calls and content
            for item in response.output:
                item_type = getattr(item, "type", None)
                
                # Handle function_call type
                if item_type == "function_call":
                    tool_calls.append({
                        "id": getattr(item, "call_id", None),
                        "type": "function",
                        "function": {
                            "name": getattr(item, "name", None),
                            "arguments": getattr(item, "arguments", "")
                        }
                    })
                
                # Handle output_text type
                elif item_type == "output_text":
                    text_content = getattr(item, "text", None)
                    if text_content:
                        content_segments.append(text_content)
                
                # Handle items with content blocks
                elif hasattr(item, "content"):
                    for block in item.content:
                        block_type = getattr(block, "type", None)
                        if block_type == "output_text":
                            text_value = getattr(block, "text", None)
                            if text_value:
                                content_segments.append(text_value)

            # Determine finish reason
            finish_reason = getattr(response, "status", None)
            if not finish_reason and hasattr(response, "output") and response.output:
                for item in response.output:
                    if hasattr(item, "finish_reason"):
                        finish_reason = item.finish_reason
                        break

            result = {
                "content": "".join(content_segments) if content_segments else None,
                "tool_calls": tool_calls if tool_calls else None,
                "finish_reason": finish_reason
            }

            return result
        except Exception as e:
            return {
                "content": f"Error in OpenAI: {str(e)}",
                "tool_calls": None,
                "finish_reason": "error"
            }
    
    def call_stream_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = 0.7,
        manage_context: bool = False,
        format: str = 'text'
    ) -> Iterator[Dict[str, Any]]:
        """
        Streams OpenAI API response with function calling (tools) support.
        
        Args:
            messages: List of messages
            tools: List of tool definitions in OpenAI format
            tool_choice: "auto", "none", or specific tool
            temperature: Temperature for generation (0-1)
            manage_context: If True, applies context management to messages before sending (default: False)
            format: Response format - 'text', 'json_object', or 'json_schema' (default: 'text')
        
        Yields:
            Dict chunks with 'type' ('content' or 'tool_call'), 'delta' (content chunk),
            'tool_call_id', 'function_name', 'function_arguments'
        """
        try:
            # Manage context if enabled
            if manage_context:
                messages = self.context_manager.manage(messages)
            
            # Build stream parameters
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "temperature": temperature,
                "stream": True
            }
            
            # Add response_format based on format parameter
            if format == 'json_object':
                params["response_format"] = {"type": "json_object"}
            elif format == 'json_schema':
                params["response_format"] = {"type": "json_schema"}
            # 'text' is the default, no need to specify
            
            stream = self.client.chat.completions.create(**params)
            
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
        verbose: bool = False,
        manage_context: bool = True
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
            manage_context: If True, applies context management
        
        Returns:
            Dict with:
                - 'content': Final LLM response
                - 'messages': Complete message history
                - 'function_calls': List of all functions called
                - 'iterations': Number of iterations performed
                - 'context_stats': Context usage statistics (if manage_context=True)
        """
        import json
        
        conversation_messages = messages.copy()
        all_function_calls = []
        iteration = 0
        
        # Track raw response for Responses API
        last_raw_response = None
        
        while iteration < max_iterations:
            iteration += 1
            
            # Manage context if enabled
            if manage_context:
                conversation_messages = self.context_manager.manage(
                    conversation_messages,
                    verbose=verbose
                )
                
                if verbose:
                    stats = self.context_manager.get_stats(conversation_messages)
                    print(f"\n📊 Context: {stats['current_tokens']} tokens ({stats['usage_percent']}% used)")
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration}")
                print(f"{'='*60}")
            
            # Call the LLM with tools and get raw response
            try:
                # Make the actual API call to get raw response
                inputs = self._normalize_messages_to_inputs(conversation_messages)

                normalized_tools = []
                
                for tool in tools:
                    if tool.get("type") == "function":
                        func_def = tool.get("function", {})
                        normalized_tools.append({
                            "type": "function",
                            "name": func_def.get("name"),
                            "description": func_def.get("description"),
                            "parameters": func_def.get("parameters"),
                        })
                
                raw_response = self.client.responses.create(
                    model=self.model,
                    input=inputs,
                    tools=normalized_tools,
                    tool_choice=tool_choice,
                    temperature=temperature
                )
                
                # Parse the response
                response = self._parse_response(raw_response)
                
                # Store raw output items for next iteration
                if hasattr(raw_response, 'output'):
                    for item in raw_response.output:
                        # Convert output items to dict format for storage
                        item_dict = {
                            "type": getattr(item, "type", None),
                        }
                        if hasattr(item, "call_id"):
                            item_dict["call_id"] = item.call_id
                        if hasattr(item, "name"):
                            item_dict["name"] = item.name
                        if hasattr(item, "arguments"):
                            item_dict["arguments"] = item.arguments
                        if hasattr(item, "text"):
                            item_dict["text"] = item.text
                        if hasattr(item, "content"):
                            item_dict["content"] = item.content
                        
                        conversation_messages.append(item_dict)
                
            except Exception as e:
                if verbose:
                    print(f"❌ Error calling API: {str(e)}")
                response = {
                    "content": f"Error: {str(e)}",
                    "tool_calls": None,
                    "finish_reason": "error"
                }
            
            # If there's content and no tool calls, we're done
            if response["content"] and not response["tool_calls"]:
                if verbose:
                    print(f"\n✅ Final response: {response['content']}")
                
                result = {
                    "content": response["content"],
                    "messages": conversation_messages,
                    "function_calls": all_function_calls,
                    "iterations": iteration,
                    "finish_reason": response["finish_reason"]
                }
                
                # Add context stats if management was enabled
                if manage_context:
                    result["context_stats"] = self.context_manager.get_stats(conversation_messages)
                
                return result
            
            # If there are no tool calls, something went wrong
            if not response["tool_calls"]:
                if verbose:
                    print("⚠️ No tool calls or content")
                
                result = {
                    "content": response.get("content") or "No response generated",
                    "messages": conversation_messages,
                    "function_calls": all_function_calls,
                    "iterations": iteration,
                    "finish_reason": response["finish_reason"]
                }
                
                if manage_context:
                    result["context_stats"] = self.context_manager.get_stats(conversation_messages)
                
                return result
            
            # Execute each tool call and collect results
            for tool_call in response["tool_calls"]:
                function_name = tool_call["function"]["name"]
                function_args_str = tool_call["function"]["arguments"]
                call_id = tool_call["id"]
                
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
                
                # Add function call output in Responses API format
                # This will be converted by _normalize_messages_to_inputs
                conversation_messages.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": function_response
                })
        
        # If we get here, we reached max_iterations
        if verbose:
            print(f"\n⚠️ Maximum iterations reached ({max_iterations})")
        
        result = {
            "content": "Maximum iterations reached without final response",
            "messages": conversation_messages,
            "function_calls": all_function_calls,
            "iterations": iteration,
            "finish_reason": "max_iterations"
        }
        
        if manage_context:
            result["context_stats"] = self.context_manager.get_stats(conversation_messages)
        
        return result
    
    def call_stream_with_function_execution(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        available_functions: Dict[str, Callable],
        tool_choice: str = "auto",
        temperature: float = 0.7,
        max_iterations: int = 5,
        verbose: bool = False,
        manage_context: bool = True
    ) -> Iterator[Dict[str, Any]]:
        """
        Streams responses while automatically handling the complete function calling cycle.
        
        This method:
        1. Streams LLM responses in real-time
        2. Detects and executes function calls
        3. Sends function results back to the LLM
        4. Continues streaming until a final response or max_iterations
        
        Args:
            messages: Initial list of messages
            tools: Tool definitions in OpenAI format
            available_functions: Dict mapping function names to callables
            tool_choice: "auto", "none", or specific tool
            temperature: Temperature for generation
            max_iterations: Maximum number of iterations to prevent infinite loops
            verbose: If True, prints debug information
            manage_context: If True, applies context management
        
        Yields:
            Dict with:
                - 'type': 'text_delta' | 'function_call_start' | 'function_call_result' | 'iteration_start' | 'final' | 'error'
                - 'content': The content based on type
                - 'iteration': Current iteration number
                - Additional fields depending on type:
                    - For 'function_call_start': 'function_name', 'arguments'
                    - For 'function_call_result': 'function_name', 'result'
                    - For 'final': 'messages', 'function_calls', 'iterations', 'context_stats'
        """
        import json
        
        conversation_messages = messages.copy()
        all_function_calls = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Manage context if enabled
            if manage_context:
                conversation_messages = self.context_manager.manage(
                    conversation_messages,
                    verbose=verbose
                )
                
                if verbose:
                    stats = self.context_manager.get_stats(conversation_messages)
                    print(f"\n📊 Context: {stats['current_tokens']} tokens ({stats['usage_percent']}% used)")
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration}")
                print(f"{'='*60}")
            
            # Notify iteration start
            yield {
                "type": "iteration_start",
                "iteration": iteration,
                "content": f"Starting iteration {iteration}"
            }
            
            # Call the LLM with tools using streaming
            try:
                inputs = self._normalize_messages_to_inputs(conversation_messages)
                
                normalized_tools = []
                for tool in tools:
                    if tool.get("type") == "function":
                        func_def = tool.get("function", {})
                        normalized_tools.append({
                            "type": "function",
                            "name": func_def.get("name"),
                            "description": func_def.get("description"),
                            "parameters": func_def.get("parameters"),
                        })
                
                # Use streaming API
                accumulated_content = []
                accumulated_tool_calls = {}
                has_content = False
                has_tool_calls = False
                
                with self.client.responses.stream(
                    model=self.model,
                    input=inputs,
                    tools=normalized_tools,
                    tool_choice=tool_choice,
                    temperature=temperature
                ) as stream:
                    for event in stream:
                        event_type = getattr(event, "type", "")
                        
                        if verbose:
                            print(f"[DEBUG] Event type: {event_type}")
                        
                        # Handle text deltas
                        if event_type == "response.output_text.delta":
                            delta_text = getattr(event, "delta", None)
                            if delta_text:
                                has_content = True
                                accumulated_content.append(delta_text)
                                yield {
                                    "type": "text_delta",
                                    "content": delta_text,
                                    "iteration": iteration
                                }
                        
                        # Handle function call item added (start of function call)
                        elif event_type == "response.output_item.added":
                            item = getattr(event, "item", None)
                            if item and getattr(item, "type", None) == "function_call":
                                has_tool_calls = True
                                output_index = getattr(event, "output_index", 0)
                                call_id = getattr(item, "call_id", None)
                                name = getattr(item, "name", "")
                                
                                if output_index not in accumulated_tool_calls:
                                    accumulated_tool_calls[output_index] = {
                                        "id": call_id,
                                        "name": name,
                                        "arguments": ""
                                    }
                                
                                if verbose:
                                    print(f"[DEBUG] Function call started: {name}")
                        
                        # Handle function call arguments delta
                        elif event_type == "response.function_call_arguments.delta":
                            output_index = getattr(event, "output_index", 0)
                            delta = getattr(event, "delta", "")
                            
                            if output_index in accumulated_tool_calls:
                                accumulated_tool_calls[output_index]["arguments"] += delta
                        
                        # Handle function call arguments done
                        elif event_type == "response.function_call_arguments.done":
                            output_index = getattr(event, "output_index", 0)
                            arguments = getattr(event, "arguments", "")
                            
                            if output_index in accumulated_tool_calls:
                                accumulated_tool_calls[output_index]["arguments"] = arguments
                                
                                if verbose:
                                    print(f"[DEBUG] Function call complete: {accumulated_tool_calls[output_index]}")
                        
                        # Handle errors
                        elif event_type in {"response.error", "response.failed"}:
                            error = getattr(event, "error", None)
                            message = getattr(error, "message", None) if error else None
                            if message:
                                yield {
                                    "type": "error",
                                    "content": f"Error in OpenAI: {message}",
                                    "iteration": iteration
                                }
                                return
                    
                    # Get final response to extract complete structure
                    final_response = stream.get_final_response()
                
                if verbose:
                    print(f"[DEBUG] Final response has output: {hasattr(final_response, 'output')}")
                    if hasattr(final_response, 'output'):
                        print(f"[DEBUG] Output items: {len(final_response.output)}")
                        for idx, item in enumerate(final_response.output):
                            print(f"[DEBUG] Item {idx} type: {getattr(item, 'type', 'unknown')}")
                
                # Extract tool calls from final response if not detected during streaming
                if hasattr(final_response, 'output') and not accumulated_tool_calls:
                    for idx, item in enumerate(final_response.output):
                        item_type = getattr(item, "type", None)
                        if item_type == "function_call":
                            call_id = getattr(item, "call_id", None)
                            if call_id:
                                has_tool_calls = True
                                accumulated_tool_calls[idx] = {
                                    "id": call_id,
                                    "name": getattr(item, "name", ""),
                                    "arguments": getattr(item, "arguments", "")
                                }
                                if verbose:
                                    print(f"[DEBUG] Found function_call in final response: {accumulated_tool_calls[idx]}")
                        elif item_type == "output_text":
                            text_content = getattr(item, "text", None)
                            if text_content and not accumulated_content:
                                has_content = True
                                accumulated_content.append(text_content)
                
                # Store the response in conversation
                if hasattr(final_response, 'output'):
                    for item in final_response.output:
                        item_dict = {
                            "type": getattr(item, "type", None),
                        }
                        if hasattr(item, "call_id"):
                            item_dict["call_id"] = item.call_id
                        if hasattr(item, "name"):
                            item_dict["name"] = item.name
                        if hasattr(item, "arguments"):
                            item_dict["arguments"] = item.arguments
                        if hasattr(item, "text"):
                            item_dict["text"] = item.text
                        if hasattr(item, "content"):
                            item_dict["content"] = item.content
                        
                        conversation_messages.append(item_dict)
                
                # If we have content but no tool calls, we're done
                if has_content and not has_tool_calls:
                    final_content = "".join(accumulated_content)
                    if verbose:
                        print(f"\n✅ Final response: {final_content}")
                    
                    result = {
                        "type": "final",
                        "content": final_content,
                        "messages": conversation_messages,
                        "function_calls": all_function_calls,
                        "iterations": iteration
                    }
                    
                    if manage_context:
                        result["context_stats"] = self.context_manager.get_stats(conversation_messages)
                    
                    yield result
                    return
                
                # If there are no tool calls, something went wrong
                if not accumulated_tool_calls:
                    if verbose:
                        print("⚠️ No tool calls or content")
                    
                    result = {
                        "type": "final",
                        "content": "".join(accumulated_content) if accumulated_content else "No response generated",
                        "messages": conversation_messages,
                        "function_calls": all_function_calls,
                        "iterations": iteration
                    }
                    
                    if manage_context:
                        result["context_stats"] = self.context_manager.get_stats(conversation_messages)
                    
                    yield result
                    return
                
                # Execute each tool call
                for output_index, tool_call_data in accumulated_tool_calls.items():
                    function_name = tool_call_data["name"]
                    function_args_str = tool_call_data["arguments"]
                    call_id = tool_call_data["id"]
                    
                    try:
                        function_args = json.loads(function_args_str)
                    except json.JSONDecodeError as e:
                        function_args = {}
                        if verbose:
                            print(f"⚠️ Error parsing arguments: {e}")
                    
                    if verbose:
                        print(f"\n🔧 Calling function: {function_name}")
                        print(f"📋 Arguments: {function_args}")
                    
                    # Notify function call start
                    yield {
                        "type": "function_call_start",
                        "function_name": function_name,
                        "arguments": function_args,
                        "iteration": iteration,
                        "content": f"Calling {function_name}"
                    }
                    
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
                    
                    # Notify function result
                    yield {
                        "type": "function_call_result",
                        "function_name": function_name,
                        "result": function_response,
                        "iteration": iteration,
                        "content": function_response
                    }
                    
                    # Add function call output to conversation
                    conversation_messages.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": function_response
                    })
                
            except Exception as e:
                if verbose:
                    print(f"❌ Error in iteration: {str(e)}")
                
                yield {
                    "type": "error",
                    "content": f"Error: {str(e)}",
                    "iteration": iteration
                }
                return
        
        # If we get here, we reached max_iterations
        if verbose:
            print(f"\n⚠️ Maximum iterations reached ({max_iterations})")
        
        result = {
            "type": "final",
            "content": "Maximum iterations reached without final response",
            "messages": conversation_messages,
            "function_calls": all_function_calls,
            "iterations": iteration,
            "finish_reason": "max_iterations"
        }
        
        if manage_context:
            result["context_stats"] = self.context_manager.get_stats(conversation_messages)
        
        yield result

    def _parse_response(self, raw_response: Any) -> Dict[str, Any]:
        """
        Parse a raw Responses API response into a standardized format
        """
        content_segments = []
        tool_calls: List[Dict[str, Any]] = []

        # Extract output_text if available
        if hasattr(raw_response, "output_text") and raw_response.output_text:
            content_segments.append(raw_response.output_text)

        # Parse output array for function calls and content
        for item in raw_response.output:
            item_type = getattr(item, "type", None)
            
            # Handle function_call type
            if item_type == "function_call":
                tool_calls.append({
                    "id": getattr(item, "call_id", None),
                    "type": "function",
                    "function": {
                        "name": getattr(item, "name", None),
                        "arguments": getattr(item, "arguments", "")
                    }
                })
            
            # Handle output_text type
            elif item_type == "output_text":
                text_content = getattr(item, "text", None)
                if text_content:
                    content_segments.append(text_content)
            
            # Handle items with content blocks
            elif hasattr(item, "content"):
                for block in item.content:
                    block_type = getattr(block, "type", None)
                    if block_type == "output_text":
                        text_value = getattr(block, "text", None)
                        if text_value:
                            content_segments.append(text_value)

        # Determine finish reason
        finish_reason = getattr(raw_response, "status", None)
        if not finish_reason and hasattr(raw_response, "output") and raw_response.output:
            for item in raw_response.output:
                if hasattr(item, "finish_reason"):
                    finish_reason = item.finish_reason
                    break

        return {
            "content": "".join(content_segments) if content_segments else None,
            "tool_calls": tool_calls if tool_calls else None,
            "finish_reason": finish_reason
        }

    def _normalize_messages_to_inputs(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Normalize messages to inputs for the OpenAI Responses API
        """
        
        # Build API call parameters for Responses API
        allowed_types = {
            "input_text",
            "input_image",
            "output_text",
            "refusal",
            "input_file",
            "computer_screenshot",
            "summary_text",
            "function_call_output"
        }

        inputs = []

        for msg in messages:
            # Handle messages that are already in Responses API format (have type but no role)
            msg_type = msg.get("type")
            
            # Pass through function_call_output and other output items directly
            if msg_type in ["function_call_output", "function_call", "output_text", "reasoning"]:
                inputs.append(msg)
                continue

            # Handle regular role-based messages
            role = msg.get("role", None)
            if not role:
                # Skip messages without role or type
                continue
                
            content = msg.get("content", "")

            if role == "assistant" or role == "tool":
                default_block_type = "output_text"
            else:
                default_block_type = "input_text"

            structured_content = []

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")

                        if block_type in allowed_types:
                            structured_content.append(block)
                            continue

                        if block_type == "text":
                            block = {
                                **block,
                                "type": default_block_type
                            }
                            structured_content.append(block)
                            continue

                        text_value = block.get("text")
                        if text_value is not None:
                            structured_content.append({
                                "type": default_block_type,
                                "text": str(text_value)
                            })
                            continue

                    structured_content.append({
                        "type": default_block_type,
                        "text": str(block)
                    })
            else:
                if content:  # Only add if content is not empty
                    structured_content.append({
                        "type": default_block_type,
                        "text": str(content)
                    })

            inputs.append({
                "role": role,
                "content": structured_content
            })

        return inputs
        
    def _normalize_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue

            tool_type = tool.get("type", "function")
            if tool_type == "function":
                function_def = tool.get("function") if isinstance(tool.get("function"), dict) else {}
                name = function_def.get("name") or tool.get("name")
                description = function_def.get("description") or tool.get("description")
                parameters = function_def.get("parameters") or tool.get("parameters")

                if not name:
                    # Skip invalid tool definitions missing a name
                    continue

                normalized.append({
                    "type": "function",
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters
                    }
                })
            else:
                normalized.append(tool)
        return normalized