# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLMeter endpoints for Google's Gemini API.

Supports Google's Generative AI API for Gemini models.

Install the dependency::

    pip install 'llmeter[gemini]'

Example:
    Direct Gemini API:

    ```python
    from llmeter.endpoints import GeminiEndpoint, GeminiStreamEndpoint

    # Non-streaming
    endpoint = GeminiEndpoint(
        model_id="gemini-2.0-flash-exp",
        api_key="your-api-key"
    )

    # Streaming
    stream_endpoint = GeminiStreamEndpoint(
        model_id="gemini-2.0-flash-exp",
        api_key="your-api-key"
    )
    ```
"""

import logging
import time
from typing import Any, Generic, Iterable, TypeVar

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse

from .base import Endpoint, InvocationResponse

logger = logging.getLogger(__name__)

TGeminiResponseBase = TypeVar(
    "TGeminiResponseBase",
    bound=GenerateContentResponse | Iterable[GenerateContentResponse],
)


class GeminiEndpointBase(Endpoint[TGeminiResponseBase], Generic[TGeminiResponseBase]):
    """Base class for Google Gemini API endpoints.

    Args:
        model_id: Gemini model identifier (e.g. ``"gemini-2.0-flash-exp"``).
        endpoint_name: Display name for this endpoint. Defaults to ``"gemini"``.
        api_key: Google API key for authentication.
        provider: Provider name. Defaults to ``"google"``.
        **kwargs: Additional keyword arguments forwarded to the model configuration
            (e.g. ``temperature``, ``top_p``, ``top_k``, ``max_output_tokens``).
    """

    def __init__(
        self,
        model_id: str,
        endpoint_name: str = "gemini",
        api_key: str | None = None,
        provider: str = "google",
        **kwargs: Any,
    ):
        super().__init__(
            endpoint_name=endpoint_name,
            model_id=model_id,
            provider=provider,
        )
        if api_key:
            genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_id)
        self._generation_config = kwargs

    def _parse_payload(self, payload: dict) -> str:
        """Extract user message text from a Gemini API payload.

        Args:
            payload: Request payload containing ``contents``.

        Returns:
            Concatenated message text content.
        """
        contents = payload.get("contents")
        if not contents:
            return ""
        
        text_parts: list[str] = []
        if isinstance(contents, str):
            return contents
        elif isinstance(contents, list):
            for content in contents:
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, dict):
                    parts = content.get("parts", [])
                    for part in parts:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
        return "\n".join(text_parts)

    @staticmethod
    def create_payload(
        user_message: str,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> dict:
        """Create a payload for the Gemini API.

        This is a convenience helper. You can also build the payload dict directly.

        Args:
            user_message: The user message text.
            max_tokens: Maximum tokens to generate. Defaults to 256.
            **kwargs: Additional generation config parameters (``temperature``,
                ``top_p``, ``top_k``, ``stop_sequences``, etc.).

        Returns:
            dict: Formatted payload for the Gemini API.

        Raises:
            ValueError: If ``max_tokens`` is not a positive integer.
            TypeError: If ``user_message`` is not a string.

        Examples:
            Text only:

            ```python
            create_payload("Hello, Gemini!")
            ```

            With generation config:

            ```python
            create_payload(
                "Explain quantum computing",
                max_tokens=1024,
                temperature=0.7,
            )
            ```
        """
        if not isinstance(user_message, str):
            raise TypeError(
                f"user_message must be a str, got {type(user_message).__name__}"
            )
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        generation_config = {"max_output_tokens": max_tokens}
        generation_config.update(kwargs)

        payload: dict = {
            "contents": user_message,
            "generation_config": generation_config,
        }
        return payload


class GeminiEndpoint(GeminiEndpointBase[GenerateContentResponse]):
    """Endpoint for Google Gemini API (non-streaming).

    Examples:
        Direct Gemini API:

        ```python
        endpoint = GeminiEndpoint(
            model_id="gemini-2.0-flash-exp",
            api_key="your-api-key"
        )
        ```

        With custom generation config:

        ```python
        endpoint = GeminiEndpoint(
            model_id="gemini-2.0-flash-exp",
            api_key="your-api-key",
            temperature=0.7,
            top_p=0.9,
        )
        ```
    """

    @GeminiEndpointBase.llmeter_invoke
    def invoke(self, payload: dict) -> GenerateContentResponse:
        """Invoke the Gemini API (non-streaming)."""
        contents = payload.get("contents", "")
        generation_config = payload.get("generation_config", {})
        
        client_response = self._model.generate_content(
            contents=contents,
            generation_config=generation_config,
            stream=False,
        )
        return client_response

    def prepare_payload(self, payload: dict, **kwargs: Any) -> dict:
        """Merge instance-level generation config with request payload."""
        prepared = {**payload}
        
        # Merge generation configs
        existing_config = prepared.get("generation_config", {})
        merged_config = {**self._generation_config, **existing_config}
        if merged_config:
            prepared["generation_config"] = merged_config
        
        return prepared

    def process_raw_response(
        self,
        raw_response: GenerateContentResponse,
        start_t: float,
        response: InvocationResponse,
    ) -> None:
        """Parse a non-streaming Gemini API response.

        Args:
            raw_response: The GenerateContentResponse object returned by the API.
            start_t: Start time of the API call.
            response: The LLMeter response object to be populated in-place.
        """
        response.time_to_last_token = time.perf_counter() - start_t
        
        # Extract text from candidates
        response_text = ""
        if raw_response.candidates:
            for candidate in raw_response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "text"):
                            response_text += part.text
        response.response_text = response_text

        # Extract token usage
        if hasattr(raw_response, "usage_metadata") and raw_response.usage_metadata:
            usage = raw_response.usage_metadata
            response.num_tokens_input = getattr(usage, "prompt_token_count", None)
            response.num_tokens_output = getattr(usage, "candidates_token_count", None)
            # Gemini supports cached content
            response.num_tokens_input_cached = getattr(
                usage, "cached_content_token_count", None
            )


class GeminiStreamEndpoint(GeminiEndpointBase[Iterable[GenerateContentResponse]]):
    """Endpoint for Google Gemini API (streaming).

    Uses streaming to enable time-to-first-token and time-to-last-token measurements.

    Args:
        model_id: Gemini model identifier.
        endpoint_name: Display name. Defaults to ``"gemini"``.
        api_key: Google API key for authentication.
        provider: Provider name. Defaults to ``"google"``.
        **kwargs: Additional arguments forwarded to generation config.

    Examples:
        Direct Gemini API:

        ```python
        endpoint = GeminiStreamEndpoint(
            model_id="gemini-2.0-flash-exp",
            api_key="your-api-key"
        )
        ```

        With custom generation config:

        ```python
        endpoint = GeminiStreamEndpoint(
            model_id="gemini-2.0-flash-exp",
            api_key="your-api-key",
            temperature=0.9,
        )
        ```
    """

    @GeminiEndpointBase.llmeter_invoke
    def invoke(self, payload: dict) -> Iterable[GenerateContentResponse]:
        """Invoke the Gemini API with streaming."""
        contents = payload.get("contents", "")
        generation_config = payload.get("generation_config", {})
        
        client_response = self._model.generate_content(
            contents=contents,
            generation_config=generation_config,
            stream=True,
        )
        return client_response

    def prepare_payload(self, payload: dict, **kwargs: Any) -> dict:
        """Merge instance-level generation config with request payload."""
        prepared = {**payload}
        
        # Merge generation configs
        existing_config = prepared.get("generation_config", {})
        merged_config = {**self._generation_config, **existing_config}
        if merged_config:
            prepared["generation_config"] = merged_config
        
        return prepared

    def process_raw_response(
        self,
        raw_response: Iterable[GenerateContentResponse],
        start_t: float,
        response: InvocationResponse,
    ) -> None:
        """Parse a streaming Gemini API response.

        Processes streaming chunks to extract text, token counts, and timing.

        Args:
            raw_response: The streaming iterator of response chunks.
            start_t: Start time of the API call.
            response: The LLMeter response object to be populated in-place.
        """
        for chunk in raw_response:
            now = time.perf_counter()

            # Extract text from the chunk
            if chunk.candidates:
                for candidate in chunk.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, "text"):
                                text = part.text
                                if text:
                                    if response.time_to_first_token is None:
                                        response.time_to_first_token = now - start_t
                                    if response.response_text is None:
                                        response.response_text = text
                                    else:
                                        response.response_text += text
                                    response.time_to_last_token = now - start_t

            # Extract token usage (typically in final chunk)
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage = chunk.usage_metadata
                response.num_tokens_input = getattr(usage, "prompt_token_count", None)
                response.num_tokens_output = getattr(
                    usage, "candidates_token_count", None
                )
                response.num_tokens_input_cached = getattr(
                    usage, "cached_content_token_count", None
                )
