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
        Based on OpenAI's official function calling documentation.
        
        Args:
            messages: List of messages in format [{"role": "user/assistant/system", "content": "..."}]
            tools: List of tool definitions in OpenAI format
            tool_choice: "auto", "none", "required", or {"type": "function", "name": "function_name"}
            temperature: Temperature for generation (0-1)
            manage_context: If True, applies context management to messages before sending (default: False)
            format: Response format - 'text', 'json_object', or 'json_schema' (default: 'text')
        
        Returns:
            Dict with:
                - 'content': Text response (str or None)
                - 'tool_calls': List of tool calls or None
                - 'output': Raw output array from API
                - 'finish_reason': Completion status
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

            # Step 2: Prompt the model with tools defined
            response = self.client.responses.create(**params)

            # Parse response according to Responses API format
            content_segments = []
            tool_calls: List[Dict[str, Any]] = []

            # Extract output_text if available
            if hasattr(response, "output_text") and response.output_text:
                content_segments.append(response.output_text)

            # Step 3: Parse output array for function calls and content
            # According to docs: response.output contains items with type="function_call"
            for item in response.output:
                item_type = getattr(item, "type", None)
                
                # Handle function_call type (as per documentation)
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
                "output": response.output if hasattr(response, "output") else [],
                "finish_reason": finish_reason
            }

            return result
        except Exception as e:
            return {
                "content": f"Error in OpenAI: {str(e)}",
                "tool_calls": None,
                "output": [],
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
        Streams OpenAI API response with function calling (tools) support using Responses API.
        Based on OpenAI's official streaming function calling documentation.
        
        Args:
            messages: List of messages
            tools: List of tool definitions in OpenAI format
            tool_choice: "auto", "none", "required", or specific tool
            temperature: Temperature for generation (0-1)
            manage_context: If True, applies context management to messages before sending (default: False)
            format: Response format - 'text', 'json_object', or 'json_schema' (default: 'text')
        
        Yields:
            Dict with streaming events:
                - 'type': Event type ('text_delta', 'tool_call_start', 'tool_call_delta', 'tool_call_done', 'final', 'error')
                - 'delta': Content delta for text or arguments
                - 'output_index': Index of the output item
                - 'tool_call': Complete tool call info (for 'tool_call_done')
                - 'content'/'tool_calls': Final accumulated data (for 'final')
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
                "temperature": temperature,
                "stream": True
            }
            
            # Add response_format based on format parameter
            if format == 'json_object':
                params["response_format"] = {"type": "json_object"}
            elif format == 'json_schema':
                params["response_format"] = {"type": "json_schema"}
            
            # Accumulate tool calls as per documentation
            accumulated_tool_calls = {}
            accumulated_content = []
            
            # Stream events from Responses API
            stream = self.client.responses.create(**params)
            
            for event in stream:
                event_type = getattr(event, "type", "")
                
                # Handle text deltas (response.output_text.delta)
                if event_type == "response.output_text.delta":
                    delta_text = getattr(event, "delta", None)
                    if delta_text:
                        accumulated_content.append(delta_text)
                        yield {
                            "type": "text_delta",
                            "delta": delta_text
                        }
                
                # Handle function call item added (response.output_item.added)
                elif event_type == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item and getattr(item, "type", None) == "function_call":
                        output_index = getattr(event, "output_index", 0)
                        call_id = getattr(item, "call_id", None)
                        name = getattr(item, "name", "")
                        
                        accumulated_tool_calls[output_index] = {
                            "id": call_id,
                            "name": name,
                            "arguments": ""
                        }
                        
                        yield {
                            "type": "tool_call_start",
                            "output_index": output_index,
                            "call_id": call_id,
                            "function_name": name
                        }
                
                # Handle function call arguments delta (response.function_call_arguments.delta)
                elif event_type == "response.function_call_arguments.delta":
                    output_index = getattr(event, "output_index", 0)
                    delta = getattr(event, "delta", "")
                    
                    if output_index in accumulated_tool_calls:
                        accumulated_tool_calls[output_index]["arguments"] += delta
                        
                        yield {
                            "type": "tool_call_delta",
                            "output_index": output_index,
                            "delta": delta
                        }
                
                # Handle function call arguments done (response.function_call_arguments.done)
                elif event_type == "response.function_call_arguments.done":
                    output_index = getattr(event, "output_index", 0)
                    arguments = getattr(event, "arguments", "")
                    
                    if output_index in accumulated_tool_calls:
                        accumulated_tool_calls[output_index]["arguments"] = arguments
                        
                        yield {
                            "type": "tool_call_done",
                            "output_index": output_index,
                            "tool_call": {
                                "id": accumulated_tool_calls[output_index]["id"],
                                "type": "function",
                                "function": {
                                    "name": accumulated_tool_calls[output_index]["name"],
                                    "arguments": arguments
                                }
                            }
                        }
                
                # Handle errors
                elif event_type in {"response.error", "response.failed"}:
                    error = getattr(event, "error", None)
                    message = getattr(error, "message", None) if error else None
                    if message:
                        yield {
                            "type": "error",
                            "delta": f"Error in OpenAI: {message}"
                        }
                        return
            
            # Yield final accumulated data
            tool_calls_list = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"]
                    }
                }
                for tc in accumulated_tool_calls.values()
            ]
            
            yield {
                "type": "final",
                "content": "".join(accumulated_content) if accumulated_content else None,
                "tool_calls": tool_calls_list if tool_calls_list else None
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "delta": f"Error in OpenAI: {str(e)}"
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
        Automatically handles the complete function calling cycle using call_with_tools.
        Based on OpenAI's official function calling documentation flow:
        
        1. Calls the LLM with tools using call_with_tools
        2. Executes the requested functions
        3. Sends the results back to the LLM
        4. Repeats until getting a final response or reaching max_iterations
        
        Args:
            messages: Initial list of messages
            tools: Tool definitions in OpenAI format
            available_functions: Dict mapping function names to callables
            tool_choice: "auto", "none", "required", or specific tool
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
        
        # Create a running input list we will add to over time (as per documentation)
        input_list = messages.copy()
        all_function_calls = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration}")
                print(f"{'='*60}")
            
            # Step 1 & 2: Prompt the model with tools defined (using call_with_tools)
            response = self.call_with_tools(
                messages=input_list,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                manage_context=manage_context
            )
            
            if verbose and manage_context:
                stats = self.context_manager.get_stats(input_list)
                print(f"\n📊 Context: {stats['current_tokens']} tokens ({stats['usage_percent']}% used)")
            
            # Save function call outputs for subsequent requests (as per documentation line 113)
            # Add the raw output items to input_list: input_list += response.output
            if response.get("output"):
                # The output contains the model's response items (function_call, output_text, etc.)
                # We need to convert them to dicts and extend input_list so the API can track call_ids
                for item in response["output"]:
                    # Convert API objects to dict format
                    item_dict = {"type": getattr(item, "type", None)}
                    
                    # Add all relevant attributes
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
                    
                    input_list.append(item_dict)
            
            # Step 5: If there's content and no tool calls, we have a final response
            if response["content"] and not response["tool_calls"]:
                if verbose:
                    print(f"\n✅ Final response: {response['content']}")
                
                result = {
                    "content": response["content"],
                    "messages": input_list,
                    "function_calls": all_function_calls,
                    "iterations": iteration,
                    "finish_reason": response["finish_reason"]
                }
                
                if manage_context:
                    result["context_stats"] = self.context_manager.get_stats(input_list)
                
                return result
            
            # If there are no tool calls, something went wrong
            if not response["tool_calls"]:
                if verbose:
                    print("⚠️ No tool calls or content")
                
                result = {
                    "content": response.get("content") or "No response generated",
                    "messages": input_list,
                    "function_calls": all_function_calls,
                    "iterations": iteration,
                    "finish_reason": response["finish_reason"]
                }
                
                if manage_context:
                    result["context_stats"] = self.context_manager.get_stats(input_list)
                
                return result
            
            # Step 3: Execute the function logic for each tool call
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
                
                # Step 4: Provide function call results to the model
                # (as per documentation format)
                input_list.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": function_response
                })
        
        # If we get here, we reached max_iterations
        if verbose:
            print(f"\n⚠️ Maximum iterations reached ({max_iterations})")
        
        result = {
            "content": "Maximum iterations reached without final response",
            "messages": input_list,
            "function_calls": all_function_calls,
            "iterations": iteration,
            "finish_reason": "max_iterations"
        }
        
        if manage_context:
            result["context_stats"] = self.context_manager.get_stats(input_list)
        
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
        Uses call_stream_with_tools internally, following OpenAI's documentation pattern.
        
        This method:
        1. Streams LLM responses in real-time using call_stream_with_tools
        2. Detects and executes function calls
        3. Sends function results back to the LLM
        4. Continues streaming until a final response or max_iterations
        
        Args:
            messages: Initial list of messages
            tools: Tool definitions in OpenAI format
            available_functions: Dict mapping function names to callables
            tool_choice: "auto", "none", "required", or specific tool
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
        
        # Create a running input list we will add to over time (as per documentation)
        input_list = messages.copy()
        all_function_calls = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration}")
                print(f"{'='*60}")
                
                if manage_context:
                    stats = self.context_manager.get_stats(input_list)
                    print(f"\n📊 Context: {stats['current_tokens']} tokens ({stats['usage_percent']}% used)")
            
            # Notify iteration start
            yield {
                "type": "iteration_start",
                "iteration": iteration,
                "content": f"Starting iteration {iteration}"
            }
            
            # Step 1 & 2: Stream the model response with tools using call_stream_with_tools
            accumulated_content = []
            accumulated_tool_calls = {}
            has_content = False
            has_tool_calls = False
            
            try:
                for event in self.call_stream_with_tools(
                    messages=input_list,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                    manage_context=manage_context
                ):
                    event_type = event.get("type")
                    
                    if verbose and event_type not in ["text_delta", "tool_call_delta"]:
                        print(f"[DEBUG] Event type: {event_type}")
                    
                    # Forward text deltas to caller
                    if event_type == "text_delta":
                        has_content = True
                        accumulated_content.append(event.get("delta", ""))
                        yield {
                            "type": "text_delta",
                            "content": event.get("delta", ""),
                            "iteration": iteration
                        }
                    
                    # Track tool call start
                    elif event_type == "tool_call_start":
                        has_tool_calls = True
                        output_index = event.get("output_index", 0)
                        accumulated_tool_calls[output_index] = {
                            "id": event.get("call_id"),
                            "name": event.get("function_name"),
                            "arguments": ""
                        }
                        
                        if verbose:
                            print(f"[DEBUG] Function call started: {event.get('function_name')}")
                    
                    # Accumulate tool call arguments
                    elif event_type == "tool_call_delta":
                        output_index = event.get("output_index", 0)
                        if output_index in accumulated_tool_calls:
                            accumulated_tool_calls[output_index]["arguments"] += event.get("delta", "")
                    
                    # Tool call complete
                    elif event_type == "tool_call_done":
                        output_index = event.get("output_index", 0)
                        tool_call = event.get("tool_call", {})
                        if output_index in accumulated_tool_calls:
                            accumulated_tool_calls[output_index]["arguments"] = tool_call.get("function", {}).get("arguments", "")
                            
                            if verbose:
                                print(f"[DEBUG] Function call complete: {accumulated_tool_calls[output_index]}")
                    
                    # Handle final event from call_stream_with_tools
                    elif event_type == "final":
                        # Update accumulated data from final event
                        if event.get("content") and not accumulated_content:
                            accumulated_content.append(event["content"])
                            has_content = True
                        
                        if event.get("tool_calls") and not accumulated_tool_calls:
                            for idx, tc in enumerate(event["tool_calls"]):
                                accumulated_tool_calls[idx] = {
                                    "id": tc.get("id"),
                                    "name": tc.get("function", {}).get("name"),
                                    "arguments": tc.get("function", {}).get("arguments", "")
                                }
                            has_tool_calls = True
                    
                    # Forward errors
                    elif event_type == "error":
                        yield {
                            "type": "error",
                            "content": event.get("delta", "Unknown error"),
                            "iteration": iteration
                        }
                        return
                
                # Save function call outputs for subsequent requests (as per documentation)
                # Build output items in the format expected by the Responses API
                if accumulated_content:
                    # Add output_text item
                    input_list.append({
                        "type": "output_text",
                        "text": "".join(accumulated_content)
                    })
                
                if accumulated_tool_calls:
                    # Add function_call items
                    for tc in accumulated_tool_calls.values():
                        input_list.append({
                            "type": "function_call",
                            "call_id": tc["id"],
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        })
                
                # Step 5: If there's content and no tool calls, we have a final response
                if has_content and not has_tool_calls:
                    final_content = "".join(accumulated_content)
                    if verbose:
                        print(f"\n✅ Final response: {final_content}")
                    
                    result = {
                        "type": "final",
                        "content": final_content,
                        "messages": input_list,
                        "function_calls": all_function_calls,
                        "iterations": iteration
                    }
                    
                    if manage_context:
                        result["context_stats"] = self.context_manager.get_stats(input_list)
                    
                    yield result
                    return
                
                # If there are no tool calls, something went wrong
                if not accumulated_tool_calls:
                    if verbose:
                        print("⚠️ No tool calls or content")
                    
                    result = {
                        "type": "final",
                        "content": "".join(accumulated_content) if accumulated_content else "No response generated",
                        "messages": input_list,
                        "function_calls": all_function_calls,
                        "iterations": iteration
                    }
                    
                    if manage_context:
                        result["context_stats"] = self.context_manager.get_stats(input_list)
                    
                    yield result
                    return
                
                # Step 3: Execute the function logic for each tool call
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
                    
                    # Step 4: Provide function call results to the model
                    input_list.append({
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
            "messages": input_list,
            "function_calls": all_function_calls,
            "iterations": iteration,
            "finish_reason": "max_iterations"
        }
        
        if manage_context:
            result["context_stats"] = self.context_manager.get_stats(input_list)
        
        yield result

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