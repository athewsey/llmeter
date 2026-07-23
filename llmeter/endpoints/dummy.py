# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A dummy endpoint that always returns "hi", useful for testing."""

import time

from .base import Endpoint, InvocationResponse


class DummyEndpoint(Endpoint[str]):
    """A dummy endpoint that always returns "hi" regardless of input.

    This is useful for testing LLMeter pipelines without calling a real LLM service.

    Example:
        ```python
        from llmeter.endpoints import DummyEndpoint

        endpoint = DummyEndpoint()
        response = endpoint.invoke(DummyEndpoint.create_payload("Hello!"))
        print(response.response_text)  # "hi"
        ```
    """

    def __init__(
        self,
        endpoint_name: str = "dummy",
        model_id: str = "dummy-model",
        provider: str = "dummy",
    ):
        """Initialize the DummyEndpoint.

        Args:
            endpoint_name: The name of the endpoint. Defaults to "dummy".
            model_id: The identifier of the model. Defaults to "dummy-model".
            provider: The provider name. Defaults to "dummy".
        """
        super().__init__(
            endpoint_name=endpoint_name,
            model_id=model_id,
            provider=provider,
        )

    @Endpoint.llmeter_invoke
    def invoke(self, payload: dict) -> str:
        """Invoke the dummy endpoint, always returning "hi".

        Args:
            payload: The input payload (ignored).

        Returns:
            The string "hi".
        """
        return "hi"

    def process_raw_response(
        self,
        raw_response: str,
        start_t: float,
        response: InvocationResponse,
    ) -> None:
        """Parse the raw response onto the InvocationResponse.

        Args:
            raw_response: The raw response string ("hi").
            start_t: The perf_counter timestamp from before the invoke call.
            response: The InvocationResponse to populate in-place.
        """
        response.response_text = raw_response
        response.time_to_first_token = time.perf_counter() - start_t
        response.time_to_last_token = time.perf_counter() - start_t
        response.num_tokens_output = 1
        response.num_tokens_input = (
            len(payload.get("prompt", "").split()) if isinstance(payload := response.input_payload, dict) else None
        )

    @staticmethod
    def create_payload(user_message: str = "", **kwargs) -> dict:
        """Create a payload for the dummy endpoint.

        Args:
            user_message: The input message (will be ignored by invoke).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            dict: A simple payload dict with the prompt.
        """
        return {"prompt": user_message}

    def _parse_payload(self, payload: dict) -> str | None:
        """Extract the input prompt from the payload.

        Args:
            payload: The prepared request payload.

        Returns:
            The prompt string, or None.
        """
        return payload.get("prompt")
