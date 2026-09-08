# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import time
from typing import Any, Generic, Sequence, TypeVar

import litellm
from litellm import CustomStreamWrapper, completion
from litellm.types.utils import ModelResponse
from litellm.utils import get_llm_provider  # type: ignore

from .base import (
    Endpoint,
    InvocationResponse,
    ReasoningType,
    delta_has_reasoning_content,
    infer_reasoning_visibility_from_model_id,
)

logger = logging.getLogger(__name__)

litellm.json_logs = True  # type: ignore
litellm.turn_off_message_logging = True
litellm.suppress_debug_info = True

os.environ["LITELLM_LOG"] = "CRITICAL"
os.environ["LITELLM_DONT_SHOW_FEEDBACK_BOX"] = "true"

TLiteLLMResponseBase = TypeVar(
    "TLiteLLMResponseBase",
    bound=CustomStreamWrapper | ModelResponse,
)


class LiteLLMBase(Endpoint[TLiteLLMResponseBase], Generic[TLiteLLMResponseBase]):
    """Base class for (streaming or non-streaming) LiteLLM-based Endpoints"""

    # Explicit typing to keep pyright happy:
    default_reasoning_visibility: ReasoningType | None

    def __init__(
        self,
        litellm_model: str,
        model_id: str | None = None,
        default_reasoning_visibility: ReasoningType | None = None,
    ):
        self.litellm_model = litellm_model
        self.default_reasoning_visibility = (
            default_reasoning_visibility
            or infer_reasoning_visibility_from_model_id(litellm_model)
        )
        model_id_inferred, provider, _, _ = get_llm_provider(litellm_model)

        logger.info(f"Using model {model_id_inferred} from provider {provider}")
        super().__init__(
            model_id=model_id or model_id_inferred,
            provider=provider,
            endpoint_name=model_id_inferred,
        )

    def _parse_payload(self, payload):
        return json.dumps(payload.get("messages"))

    @staticmethod
    def create_payload(
        user_message: str | Sequence[str],
        max_tokens: int = 256,
        system_message: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """
        Create a payload for the LiteLLM `completion()` request.

        Args:
            user_message (str | Sequence[str]): The user's message or a sequence of messages.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 256.
            **kwargs: Additional keyword arguments to include in the payload.

        Returns:
            dict: The formatted payload for the Bedrock API request.
        """

        if isinstance(user_message, str):
            user_message = [user_message]
        payload = {
            "messages": [{"role": "user", "content": k} for k in user_message],
            "max_tokens": max_tokens,
        }
        payload.update(kwargs)
        if system_message:
            payload["messages"].append({"role": "system", "content": system_message})
        return payload


class LiteLLM(LiteLLMBase[ModelResponse]):
    """Endpoint for LiteLLM SDK-based models (non-streaming mode)

    Args:
        litellm_model: The LiteLLM model string, e.g. `"anthropic/claude-sonnet-4-6"`.
        model_id: Override for the reported model ID. Inferred from `litellm_model` if omitted.
        default_reasoning_visibility: What to record as
            [`reasoning_type`][llmeter.endpoints.base.InvocationResponse] when reasoning content is
            present. LiteLLM normalizes reasoning from many providers onto the same fields, so the
            stream carries no indication of whether it is verbatim or summarized. Defaults to a
            guess from `litellm_model` via
            [`infer_reasoning_visibility_from_model_id`][llmeter.endpoints.base.infer_reasoning_visibility_from_model_id],
            which reads the provider prefix - so `"anthropic/..."` and `"bedrock/anthropic...."`
            resolve to `"summary"`, and other providers to `"verbatim"`. Pass `"unknown"` to decline
            guessing.
    """

    @LiteLLMBase.llmeter_invoke
    def invoke(self, payload) -> ModelResponse:
        # In non-streaming mode, completion always returns a ModelResponse:
        return completion(**payload)  # type: ignore

    def prepare_payload(self, payload: dict) -> dict:
        # Make a copy of payload to avoid modifying the original
        payload_copy = payload.copy()
        # Ensure correct model ID
        payload_copy["model"] = self.litellm_model
        # Ensure streaming is disabled
        payload_copy["stream"] = False
        return payload_copy

    def process_raw_response(
        self, raw_response, start_t: float, response: InvocationResponse
    ) -> None:
        response.time_to_last_token = time.perf_counter() - start_t
        response.id = raw_response.id

        try:
            usage = raw_response.usage  # type: ignore
            response.num_tokens_input = usage.prompt_tokens
            response.num_tokens_output = usage.completion_tokens
        except AttributeError:
            pass

        message = raw_response.choices[0].message
        response.response_text = message.content
        if delta_has_reasoning_content(message):
            response.reasoning_type = self.default_reasoning_visibility or "unknown"


class LiteLLMStreaming(LiteLLMBase[CustomStreamWrapper]):
    """Streaming endpoint for any model reachable through LiteLLM.

    Args:
        litellm_model: The LiteLLM model string, e.g. `"anthropic/claude-sonnet-4-6"`.
        model_id: Override for the reported model ID. Inferred from `litellm_model` if omitted.
        default_reasoning_visibility: What to record as
            [`reasoning_type`][llmeter.endpoints.base.InvocationResponse] when reasoning content is
            present. LiteLLM normalizes reasoning from many providers onto the same fields, so the
            stream carries no indication of whether it is verbatim or summarized. Defaults to a
            guess from `litellm_model` via
            [`infer_reasoning_visibility_from_model_id`][llmeter.endpoints.base.infer_reasoning_visibility_from_model_id],
            which reads the provider prefix - so `"anthropic/..."` and `"bedrock/anthropic...."`
            resolve to `"summary"`, and other providers to `"verbatim"`. Pass `"unknown"` to decline
            guessing.
    """

    def __init__(
        self,
        litellm_model: str,
        model_id: str | None = None,
        default_reasoning_visibility: ReasoningType | None = None,
    ):
        super().__init__(
            litellm_model=litellm_model,
            model_id=model_id,
            default_reasoning_visibility=default_reasoning_visibility,
        )

    @LiteLLMBase.llmeter_invoke
    def invoke(self, payload) -> CustomStreamWrapper:
        # In streaming mode, completion always returns a CustomStreamWrapper:
        return completion(**payload)  # type: ignore

    def prepare_payload(self, payload):
        # Make a copy of payload to avoid modifying the original
        payload_copy = payload.copy()

        # Ensure correct model ID
        payload_copy["model"] = self.litellm_model

        # Ensure streaming is enabled
        payload_copy["stream"] = True

        # Ensure stream_options includes usage
        existing_options = payload_copy.get("stream_options", {})
        payload_copy["stream_options"] = {**existing_options, "include_usage": True}

        return payload_copy

    def process_raw_response(
        self,
        raw_response: CustomStreamWrapper,
        start_t: float,
        response: InvocationResponse,
    ) -> None:
        """Parse a streaming LiteLLM response.

        LiteLLM normalizes provider reasoning/thinking output onto `delta.reasoning_content` and
        `delta.thinking_blocks`. Those chunks set `time_to_first_token` but never contribute to
        `response_text`; the first chunk with visible `delta.content` sets
        `time_to_first_content_token`.
        """
        usage = None
        got_chunk_id = False
        saw_reasoning = False

        for chunk in raw_response:
            now = time.perf_counter()

            if not got_chunk_id and chunk.id is not None:
                response.id = chunk.id
                got_chunk_id = True

            # The final usage-only chunk can carry an empty `choices` list:
            choices = getattr(chunk, "choices", None)
            if choices:
                delta = choices[0].delta
                content = getattr(delta, "content", None) or ""
                if content:
                    if response.time_to_first_token is None:
                        response.time_to_first_token = now - start_t
                    if response.time_to_first_content_token is None:
                        response.time_to_first_content_token = now - start_t
                    if response.response_text is None:
                        response.response_text = content
                    else:
                        response.response_text += content
                    response.time_to_last_token = now - start_t
                elif delta_has_reasoning_content(delta):
                    saw_reasoning = True
                    if response.time_to_first_token is None:
                        response.time_to_first_token = now - start_t

            try:
                usage = chunk.usage  # type: ignore
            except AttributeError:
                continue

        if saw_reasoning:
            response.reasoning_type = self.default_reasoning_visibility or "unknown"

        if usage:
            response.num_tokens_input = usage.prompt_tokens
            response.num_tokens_output = usage.completion_tokens
