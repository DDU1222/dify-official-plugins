from typing import Any, Mapping, Optional, Sequence, Union, Tuple, Generator, Callable
import logging

from openai import OpenAI
from httpx import Timeout
from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageContentType,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)

logger = logging.getLogger(__name__)


class AihubmixOpenAIResponses:
    def __init__(self, credentials: Mapping[str, Any]):
        self.client = OpenAI(**self._to_credential_kwargs(credentials))
        self.credentials = dict(credentials)

    def _to_credential_kwargs(self, credentials: Mapping[str, Any]) -> Mapping[str, Any]:
        # Align with aihubmix provider style (see anthropic.py)
        return {
            "api_key": credentials["api_key"],
            "base_url": "https://aihubmix.com/v1",
            "timeout": Timeout(315.0, read=300.0, write=10.0, connect=5.0),
            "max_retries": 1,
        }

    def _convert_messages_to_responses_input(self, prompt_messages: Sequence[PromptMessage]) -> str:
        role_map = {
            UserPromptMessage: "user",
            AssistantPromptMessage: "assistant",
            ToolPromptMessage: "tool",
        }
        input_parts: list[str] = []
        for m in prompt_messages:
            role = role_map.get(type(m))
            if not role:
                continue

            content_str = ""
            if isinstance(m.content, str):
                content_str = m.content
            elif isinstance(m.content, list):
                content_str = "\n".join(
                    [item.data for item in m.content if item.type == PromptMessageContentType.TEXT]
                )

            if content_str:
                input_parts.append(f"{role}: {content_str}")
        return "\n\n".join(input_parts)

    def _prepare_params(self, model_parameters: Mapping[str, Any], user: Optional[str] = None) -> dict:
        """准备 Responses API 参数，处理思考模式等特殊参数"""
        params = dict(model_parameters)
        
        # 转换 max_completion_tokens -> max_output_tokens
        if "max_completion_tokens" in params:
            params["max_output_tokens"] = params.pop("max_completion_tokens")
        
        # 处理思考模式
        enable_thinking = params.pop("_enable_thinking", False)
        
        # 构建 reasoning 参数
        reasoning_effort = params.pop("reasoning_effort", None)
        reasoning_config = {}
        
        if reasoning_effort:
            reasoning_config["effort"] = reasoning_effort
        
        # 判断是否需要启用思考：enable_thinking=true 或 reasoning_effort 不是 "none"
        should_enable_thinking = enable_thinking or (reasoning_effort and reasoning_effort != "none")
        
        if should_enable_thinking:
            # 添加 include 参数以获取加密的思考内容
            params["include"] = ["reasoning.encrypted_content"]
            # 如果没有设置 effort，默认使用 medium
            if "effort" not in reasoning_config:
                reasoning_config["effort"] = "medium"
            # 添加 summary 配置
            reasoning_config["summary"] = "detailed"
        
        # 只有当 reasoning_config 非空时才添加
        if reasoning_config:
            params["reasoning"] = reasoning_config
        
        # 处理 verbosity 参数，放入 text 对象
        verbosity = params.pop("verbosity", None)
        if verbosity:
            params["text"] = {"verbosity": verbosity}
        
        if user:
            params["user"] = user
            
        return params

    def create_raw(
        self,
        *,
        model: str,
        prompt_messages: Sequence[PromptMessage],
        model_parameters: Mapping[str, Any],
        user: Optional[str] = None,
    ) -> Tuple[Any, str]:
        params = self._prepare_params(model_parameters, user)
        final_input = self._convert_messages_to_responses_input(prompt_messages)
        logger.info(f"Aihubmix Responses API Request: model={model} params={params}")

        resp_obj = self.client.responses.create(
            model=model,
            input=final_input,
            extra_headers={"APP-Code": "Dify2025"},
            **params
        )
        
        # 处理思考内容
        text_content = ""
        reasoning_content = ""
        
        # 尝试从 output 中提取思考内容
        if hasattr(resp_obj, "output") and resp_obj.output:
            for item in resp_obj.output:
                if hasattr(item, "type"):
                    if item.type == "reasoning" and hasattr(item, "content"):
                        # 提取思考内容
                        for content_item in (item.content or []):
                            if hasattr(content_item, "text"):
                                reasoning_content += content_item.text or ""
                    elif item.type == "message" and hasattr(item, "content"):
                        # 提取普通内容
                        for content_item in (item.content or []):
                            if hasattr(content_item, "text"):
                                text_content += content_item.text or ""
        
        # 如果没有从 output 获取到，尝试 output_text
        if not text_content:
            text_content = resp_obj.output_text or ""
        
        # 组合思考内容和普通内容
        if reasoning_content:
            text_content = f"<think>\n{reasoning_content}\n</think>{text_content}"
        
        return resp_obj, text_content

    def stream_raw(
        self,
        *,
        model: str,
        prompt_messages: Sequence[PromptMessage],
        model_parameters: Mapping[str, Any],
        user: Optional[str] = None,
    ) -> Generator[Tuple[str, Mapping[str, Any]], None, None]:
        """
        Yield tuple(kind, payload):
        - ("reasoning_start", {}) when reasoning begins
        - ("reasoning_delta", {"text": str}) for incremental reasoning text
        - ("reasoning_end", {}) when reasoning ends
        - ("delta", {"text": str}) for incremental output text
        - ("final", {"response": Response, "text": str}) at completion
        
        GPT-5 Responses API SSE 事件类型：
        - response.output_item.added: 添加输出项 (type=reasoning 或 type=message)
        - response.reasoning_summary_part.added: 开始思考摘要
        - response.reasoning_summary_text.delta: 思考摘要增量文本
        - response.reasoning_summary_text.done: 思考摘要完成
        - response.reasoning_summary_part.done: 思考摘要部分完成
        - response.output_item.done: 输出项完成
        - response.content_part.added: 开始内容部分
        - response.output_text.delta: 输出文本增量
        - response.output_text.done: 输出文本完成
        - response.content_part.done: 内容部分完成
        - response.completed: 响应完成
        """
        params = self._prepare_params(model_parameters, user)
        final_input = self._convert_messages_to_responses_input(prompt_messages)
        logger.info(f"Aihubmix Responses API Stream Request: model={model} params={params}")

        is_reasoning = False
        
        with self.client.responses.stream(
            model=model,
            input=final_input,
            extra_headers={"APP-Code": "Dify2025"},
            **params
        ) as stream:
            for event in stream:
                etype = getattr(event, "type", None)
                
                # 处理思考摘要开始事件
                if etype == "response.reasoning_summary_part.added":
                    if not is_reasoning:
                        is_reasoning = True
                        yield ("reasoning_start", {})
                
                # 处理思考摘要增量事件 (GPT-5 系列使用这个事件类型)
                elif etype == "response.reasoning_summary_text.delta":
                    delta_text = getattr(event, "delta", "") or ""
                    if delta_text:
                        if not is_reasoning:
                            is_reasoning = True
                            yield ("reasoning_start", {})
                        yield ("reasoning_delta", {"text": delta_text})
                
                # 处理思考摘要完成事件
                elif etype == "response.reasoning_summary_text.done":
                    # 不在这里结束思考，等待 output_text.delta 或 response.completed
                    pass
                
                elif etype == "response.reasoning_summary_part.done":
                    # 思考摘要部分完成，但不立即结束思考状态
                    pass
                
                # 处理旧版思考内容事件 (兼容性)
                elif etype == "response.reasoning_content.delta":
                    delta_text = getattr(event, "delta", "") or ""
                    if delta_text:
                        if not is_reasoning:
                            is_reasoning = True
                            yield ("reasoning_start", {})
                        yield ("reasoning_delta", {"text": delta_text})
                
                elif etype == "response.reasoning_content.done":
                    if is_reasoning:
                        is_reasoning = False
                        yield ("reasoning_end", {})
                
                # 处理加密思考内容事件 (兼容性)
                elif etype == "response.reasoning.delta":
                    delta_text = getattr(event, "delta", "") or ""
                    if delta_text:
                        if not is_reasoning:
                            is_reasoning = True
                            yield ("reasoning_start", {})
                        yield ("reasoning_delta", {"text": delta_text})
                
                elif etype == "response.reasoning.done":
                    if is_reasoning:
                        is_reasoning = False
                        yield ("reasoning_end", {})
                
                # 处理输出文本增量事件
                elif etype == "response.output_text.delta":
                    # 如果还在思考状态，先结束思考
                    if is_reasoning:
                        is_reasoning = False
                        yield ("reasoning_end", {})
                    delta_text = getattr(event, "delta", "") or ""
                    if delta_text:
                        yield ("delta", {"text": delta_text})
                
                elif etype == "response.completed":
                    # 确保思考状态已关闭
                    if is_reasoning:
                        yield ("reasoning_end", {})
                    final = stream.get_final_response()
                    full_text = getattr(final, "output_text", None) or ""
                    yield ("final", {"response": final, "text": full_text})
                    break
                
                elif etype == "response.error":
                    err = getattr(event, "error", None)
                    message = (getattr(err, "message", None) or str(err)) if err else "Responses stream error"
                    raise RuntimeError(message)

    def create_llm_result(
        self,
        *,
        model: str,
        prompt_messages: Sequence[PromptMessage],
        model_parameters: Mapping[str, Any],
        compute_usage: Callable[[int, int], Mapping[str, Any]],
        user: Optional[str] = None,
    ) -> LLMResult:
        resp_obj, text_content = self.create_raw(
            model=model,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters,
            user=user,
        )
        assistant_prompt_message = AssistantPromptMessage(content=text_content)

        prompt_tokens = 0
        completion_tokens = 0
        if getattr(resp_obj, "usage", None):
            prompt_tokens = getattr(resp_obj.usage, "input_tokens", 0) or 0
            completion_tokens = getattr(resp_obj.usage, "output_tokens", 0) or 0

        usage = compute_usage(prompt_tokens, completion_tokens)

        result = LLMResult(
            model=getattr(resp_obj, "model", model),
            prompt_messages=list(prompt_messages),
            message=assistant_prompt_message,
            usage=usage,
            system_fingerprint=None,
        )
        return result

    def stream_llm_chunks(
        self,
        *,
        model: str,
        prompt_messages: Sequence[PromptMessage],
        model_parameters: Mapping[str, Any],
        compute_usage: Callable[[int, int], Mapping[str, Any]],
        user: Optional[str] = None,
    ) -> Generator[LLMResultChunk, None, None]:
        full_text = ""
        index = 0
        final_response = None
        
        for kind, payload in self.stream_raw(
            model=model,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters,
            user=user,
        ):
            if kind == "reasoning_start":
                # 开始思考，输出 <think> 标签
                delta_text = "<think>\n"
                full_text += delta_text
                yield LLMResultChunk(
                    model=model,
                    prompt_messages=prompt_messages,
                    delta=LLMResultChunkDelta(
                        index=index,
                        message=AssistantPromptMessage(content=delta_text),
                    ),
                )
                index += 1
            
            elif kind == "reasoning_delta":
                # 思考内容增量
                delta_text = payload.get("text", "")
                if not delta_text:
                    continue
                full_text += delta_text
                yield LLMResultChunk(
                    model=model,
                    prompt_messages=prompt_messages,
                    delta=LLMResultChunkDelta(
                        index=index,
                        message=AssistantPromptMessage(content=delta_text),
                    ),
                )
                index += 1
            
            elif kind == "reasoning_end":
                # 结束思考，输出 </think> 标签
                delta_text = "\n</think>"
                full_text += delta_text
                yield LLMResultChunk(
                    model=model,
                    prompt_messages=prompt_messages,
                    delta=LLMResultChunkDelta(
                        index=index,
                        message=AssistantPromptMessage(content=delta_text),
                    ),
                )
                index += 1
            
            elif kind == "delta":
                # 普通内容增量
                delta_text = payload.get("text", "")
                if not delta_text:
                    continue
                full_text += delta_text
                yield LLMResultChunk(
                    model=model,
                    prompt_messages=prompt_messages,
                    delta=LLMResultChunkDelta(
                        index=index,
                        message=AssistantPromptMessage(content=delta_text),
                    ),
                )
                index += 1
            
            elif kind == "final":
                final_response = payload.get("response")
                break

        prompt_tokens = 0
        completion_tokens = 0
        if final_response and getattr(final_response, "usage", None):
            prompt_tokens = getattr(final_response.usage, "input_tokens", 0) or 0
            completion_tokens = getattr(final_response.usage, "output_tokens", 0) or 0

        usage = compute_usage(prompt_tokens, completion_tokens)

        yield LLMResultChunk(
            model=model,
            prompt_messages=prompt_messages,
            delta=LLMResultChunkDelta(
                index=index,
                message=AssistantPromptMessage(content=""),
                finish_reason="stop",
                usage=usage,
            ),
        )
