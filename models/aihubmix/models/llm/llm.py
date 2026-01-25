import codecs
import logging
from collections.abc import Generator
from typing import Optional, Union, Any, cast

import requests
from pydantic import TypeAdapter, ValidationError

from dify_plugin.entities.model import AIModelEntity
from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta
from dify_plugin.entities.model.message import PromptMessage, PromptMessageTool, AssistantPromptMessage
from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageTool,
)
from dify_plugin.errors.model import (
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model.openai_compatible.llm import _increase_tool_call
from .anthropic import AnthropicLargeLanguageModel
from .google import GoogleLargeLanguageModel
from .openai_response import AihubmixOpenAIResponses

# 如果两个类都继承自同一个基类，可以使用相同的初始化方式
model_schemas = []  # 或者从某处获取适当的模型模式
anthropic_llm = AnthropicLargeLanguageModel(model_schemas)
google_llm = GoogleLargeLanguageModel(model_schemas)
logger = logging.getLogger(__name__)

# thinking models compatibility for max_completion_tokens (all starting with "o" or "gpt-5")
THINKING_SERIES_COMPATIBILITY = ("o", "gpt-5")
# GPT-5 系列和 o3-pro 使用 Responses API
RESPONSE_SERIES_COMPATIBILITY = ("gpt-5", "o3-pro")


class AihubmixLargeLanguageModel(OAICompatLargeLanguageModel):
    def _update_credential(self, model: str, credentials: dict):
        credentials["endpoint_url"] = "https://aihubmix.com/v1"
        credentials["mode"] = self.get_model_mode(model).value
        credentials["function_calling_type"] = "tool_call"
        credentials["extra_headers"] = {
            "APP-Code": "Dify2025"
        }

    def _wrap_thinking_by_reasoning_content(
        self, delta: dict, is_reasoning: bool
    ) -> tuple[str, bool]:
        """
        If the reasoning response is from delta.get("reasoning") or delta.get("reasoning_content"),
        we wrap it with HTML think tag.

        :param delta: delta dictionary from LLM streaming response
        :param is_reasoning: is reasoning
        :return: tuple of (processed_content, is_reasoning)
        """
        content = delta.get("content") or ""
        # Support both "reasoning" and "reasoning_content" field names for compatibility
        reasoning_content = delta.get("reasoning") or delta.get("reasoning_content")

        if reasoning_content:
            if not is_reasoning:
                content = "<think>\n" + reasoning_content
                is_reasoning = True
            else:
                content = reasoning_content
        elif is_reasoning and content:
            content = "\n</think>" + content
            is_reasoning = False
        return content, is_reasoning

    def _handle_generate_stream_response(
        self,
        model: str,
        credentials: dict,
        response: requests.Response,
        prompt_messages: list[PromptMessage],
    ) -> Generator:
        """
        Handle llm stream response with reasoning/thinking content support.

        This method is based on openrouter's implementation to properly handle
        the reasoning_content field returned by GPT-5 series and other reasoning models.

        :param model: model name
        :param credentials: model credentials
        :param response: streamed response
        :param prompt_messages: prompt messages
        :return: llm response chunk generator
        """
        chunk_index = 0
        full_assistant_content = ""
        tools_calls: list[AssistantPromptMessage.ToolCall] = []
        finish_reason = None
        usage = None
        is_reasoning_started = False

        # delimiter for stream response, need unicode_escape
        delimiter = credentials.get("stream_mode_delimiter", "\n\n")
        delimiter = codecs.decode(delimiter, "unicode_escape")

        for chunk in response.iter_lines(decode_unicode=True, delimiter=delimiter):
            chunk = chunk.strip()
            if chunk:
                # ignore sse comments
                if chunk.startswith(":"):
                    continue
                decoded_chunk = chunk.strip().removeprefix("data:").lstrip()
                if decoded_chunk == "[DONE]":
                    continue

                try:
                    chunk_json: dict = TypeAdapter(dict[str, Any]).validate_json(decoded_chunk)
                except ValidationError:
                    yield self._create_final_llm_result_chunk(
                        index=chunk_index + 1,
                        message=AssistantPromptMessage(content=""),
                        finish_reason="Non-JSON encountered.",
                        usage=usage,
                        model=model,
                        credentials=credentials,
                        prompt_messages=prompt_messages,
                        full_content=full_assistant_content,
                    )
                    break

                # handle the error
                if chunk_json.get("error") and chunk_json.get("choices") is None:
                    raise ValueError(chunk_json.get("error"))

                if chunk_json:
                    if u := chunk_json.get("usage"):
                        usage = u
                if not chunk_json or len(chunk_json.get("choices", [])) == 0:
                    continue

                choice = chunk_json["choices"][0]
                finish_reason = choice.get("finish_reason")
                chunk_index += 1

                if "delta" in choice:
                    delta = choice["delta"]
                    # Process reasoning content with _wrap_thinking_by_reasoning_content
                    delta_content, is_reasoning_started = self._wrap_thinking_by_reasoning_content(
                        delta, is_reasoning_started
                    )

                    assistant_message_tool_calls = None

                    if (
                        "tool_calls" in delta
                        and credentials.get("function_calling_type", "no_call") == "tool_call"
                    ):
                        assistant_message_tool_calls = delta.get("tool_calls", None)
                    elif (
                        "function_call" in delta
                        and credentials.get("function_calling_type", "no_call") == "function_call"
                    ):
                        assistant_message_tool_calls = [
                            {
                                "id": "tool_call_id",
                                "type": "function",
                                "function": delta.get("function_call", {}),
                            }
                        ]

                    # extract tool calls from response
                    if assistant_message_tool_calls:
                        tool_calls = self._extract_response_tool_calls(assistant_message_tool_calls)
                        _increase_tool_call(tool_calls, tools_calls)

                    if not delta_content:
                        continue

                    # transform assistant message to prompt message
                    assistant_prompt_message = AssistantPromptMessage(content=delta_content)
                    full_assistant_content += delta_content

                elif "text" in choice:
                    choice_text = choice.get("text", "")
                    if choice_text == "":
                        continue

                    assistant_prompt_message = AssistantPromptMessage(content=choice_text)
                    full_assistant_content += choice_text
                else:
                    continue

                yield LLMResultChunk(
                    model=model,
                    prompt_messages=prompt_messages,
                    delta=LLMResultChunkDelta(index=chunk_index, message=assistant_prompt_message),
                )

            chunk_index += 1

        if tools_calls:
            yield LLMResultChunk(
                model=model,
                prompt_messages=prompt_messages,
                delta=LLMResultChunkDelta(
                    index=chunk_index,
                    message=AssistantPromptMessage(tool_calls=tools_calls, content=""),
                ),
            )

        yield self._create_final_llm_result_chunk(
            index=chunk_index,
            message=AssistantPromptMessage(content=""),
            finish_reason=finish_reason,
            usage=usage,
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            full_content=full_assistant_content,
        )

    def _prepare_model_parameters(
        self,
        model: str,
        model_parameters: dict
    ) -> dict:
        params = dict(model_parameters)

        # Claude and RESPONSE_SERIES models do not require any parameter mapping
        if (
            model.startswith("claude")
            or model.startswith(RESPONSE_SERIES_COMPATIBILITY)
        ):
            return params

        # Nothing to do if max_tokens is not provided
        if "max_tokens" not in params:
            logger.warning(f"max_tokens not found in params, using default behavior. params=%s", params)
            return params

        # For THINKING_SERIES, max_tokens always takes precedence and overwrites
        if model.startswith(THINKING_SERIES_COMPATIBILITY):
            params["max_completion_tokens"] = params.pop("max_tokens")

        # For other models, only map if max_completion_tokens is not already set
        elif "max_completion_tokens" not in params:
            params["max_completion_tokens"] = params.pop("max_tokens")

        return params

    def _dispatch_to_appropriate_model(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None
    ) -> Union[LLMResult, Generator]:
        """根据模型名称分发到适当的模型处理类"""
        # 预处理模型参数
        model_parameters = self._prepare_model_parameters(model, model_parameters)
        
        # 检查模型名称是否以 "claude" 开头
        if model.startswith("claude"):
            return anthropic_llm._invoke(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)
        
        # 检查模型名称是否以 "gemini" 开头且不以 "-nothink" 或 "-search" 结尾
        if model.startswith("gemini") and not (model.endswith("-nothink") or model.endswith("-search")):
            return google_llm._invoke(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)
                
        # 走 response 接口，其他模型走 generate 接口
        if model.startswith(RESPONSE_SERIES_COMPATIBILITY):
            # 使用 Responses API（委托给 openai_response 封装；支持流式/非流式）
            resp_handler = AihubmixOpenAIResponses(credentials)
            compute_usage = lambda pt, ct: self._calc_response_usage(
                model=model,
                credentials=credentials,
                prompt_tokens=pt,
                completion_tokens=ct,
            )
            if stream:
                return resp_handler.stream_llm_chunks(
                    model=model,
                    prompt_messages=prompt_messages,
                    model_parameters=model_parameters,
                    compute_usage=compute_usage,
                    user=user,
                )

            return resp_handler.create_llm_result(
                model=model,
                prompt_messages=prompt_messages,
                model_parameters=model_parameters,
                compute_usage=compute_usage,
                user=user,
            )
        
        # 默认使用父类的生成方法
        return super()._generate(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        try:
            self._update_credential(model, credentials)
            
            # 处理 enable_thinking 参数
            enable_thinking = model_parameters.pop("enable_thinking", None)
            
            # 对于 RESPONSE_SERIES 模型，保留 enable_thinking 用于 Responses API
            if model.startswith(RESPONSE_SERIES_COMPATIBILITY):
                if enable_thinking is not None:
                    model_parameters["_enable_thinking"] = bool(enable_thinking)
            else:
                # 对于其他模型，转换为 chat_template_kwargs
                if enable_thinking is not None:
                    model_parameters["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}

            # 将自定义的 enable_stream 参数映射到本地 stream 标志，避免把未知参数透传给上游
            enable_stream = model_parameters.pop("enable_stream", None)
            if enable_stream is not None:
                stream = bool(enable_stream)

            return self._dispatch_to_appropriate_model(
                model, credentials, prompt_messages, model_parameters, tools, stop, stream, user
            )
        except Exception as e:
            # 记录异常信息
            logger.error(f"Error invoking model {model}: {str(e)}")
            
            # 根据异常类型映射到统一的错误类型
            for error_type, exception_types in self._invoke_error_mapping.items():
                if any(isinstance(e, exc_type) for exc_type in exception_types):
                    raise error_type(str(e))
            
            # 如果没有匹配的错误类型，则抛出原始异常
            raise InvokeError(f"Unexpected error: {str(e)}")

    def validate_credentials(self, model: str, credentials: dict) -> None:
        self._update_credential(model, credentials)
        
        if model.startswith(THINKING_SERIES_COMPATIBILITY):
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=credentials.get("api_key"),
                    base_url=credentials.get("endpoint_url"),
                    timeout=10.0,
                    max_retries=1,
                )
                
                models = client.models.list()
                return
            except Exception as e:
                if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                    logger.warning(f"Ignoring expected validation error: {e}")
                    return
                else:
                    raise InvokeAuthorizationError(f"Credentials validation failed: {str(e)}")
        else:
            return super().validate_credentials(model, credentials)

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        self._update_credential(model, credentials)
        return super().get_customizable_model_schema(model, credentials)

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        self._update_credential(model, credentials)
        return super().get_num_tokens(model, credentials, prompt_messages, tools)
    
    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the error type thrown to the caller
        The value is the error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke error mapping
        """
        return {
            InvokeConnectionError: [Exception],
            InvokeServerUnavailableError: [Exception],
            InvokeRateLimitError: [Exception],
            InvokeAuthorizationError: [Exception],
            InvokeBadRequestError: [Exception],
        }