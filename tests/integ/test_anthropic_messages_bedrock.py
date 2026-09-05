# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Anthropic Messages API endpoints via Amazon Bedrock Mantle.

This module verifies that the llmeter AnthropicMessages and AnthropicMessagesStream
wrappers work correctly with the Anthropic Messages API served through the
Bedrock Mantle endpoint (``bedrock-mantle.{region}.api.aws/anthropic/v1/messages``).

Tests are marked with @pytest.mark.integ and are skipped by default to avoid
AWS costs and credential requirements during regular development.

To run these tests:
    uv run pytest tests/integ/test_anthropic_messages_bedrock.py -m integ

Required AWS Permissions:
    - bedrock-mantle:CreateInference (or equivalent Bedrock permissions)

Environment Variables:
    - BEDROCK_ANTHROPIC_MANTLE_REGION: AWS region (default: us-east-1)
    - BEDROCK_ANTHROPIC_MANTLE_TEST_MODEL: Model ID
      (default: anthropic.claude-opus-4-7)

Estimated Cost:
    - ~$0.001 per test run (Opus 4.7 pricing)
    - ~$0.003 total for all tests in this module
"""

from datetime import datetime

import pytest

from llmeter.endpoints.anthropic_messages import (
    AnthropicMessages,
    AnthropicMessagesEndpoint,
    AnthropicMessagesStream,
)


@pytest.mark.integ
def test_anthropic_messages_non_streaming(
    aws_credentials,
    bedrock_anthropic_mantle_region,
    bedrock_anthropic_mantle_test_model,
):
    """
    Test AnthropicMessages endpoint with Bedrock Mantle (non-streaming).

    Validates that the endpoint can:
    - Initialize with Bedrock Mantle provider and AWS credentials
    - Invoke the Anthropic Messages API via create_payload helper
    - Return an InvocationResponse with text, token counts, and timing
    - Complete without errors

    Args:
        aws_credentials: Boto3 session with valid AWS credentials.
        bedrock_anthropic_mantle_region: AWS region for Bedrock Mantle.
        bedrock_anthropic_mantle_test_model: Anthropic model ID for Bedrock Mantle.

    Estimated Cost: ~$0.001 per run
    """
    endpoint = AnthropicMessages(
        model_id=bedrock_anthropic_mantle_test_model,
        provider="bedrock-mantle",
        aws_region=bedrock_anthropic_mantle_region,
    )

    payload = AnthropicMessagesEndpoint.create_payload(
        user_message="Hello, this is a test. Please respond with a brief greeting.",
        max_tokens=100,
    )

    response = endpoint.invoke(payload)

    # No errors
    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )

    # Response text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert isinstance(response.response_text, str)

    # Token counts
    assert response.num_tokens_input is not None, "Input token count should be present"
    assert response.num_tokens_input > 0, "Input token count should be positive"
    assert response.num_tokens_output is not None, (
        "Output token count should be present"
    )
    assert response.num_tokens_output > 0, "Output token count should be positive"

    # Timing (back-filled by base class for non-streaming)
    assert response.time_to_last_token is not None, "Response time should be present"
    assert response.time_to_last_token > 0, "Response time should be positive"

    # Response ID (Anthropic msg_ format)
    assert response.id is not None, "Response should have an ID"
    assert response.id.startswith("msg_"), (
        f"Response ID should be Anthropic format (msg_...), got: {response.id}"
    )

    # Metadata back-fill
    assert isinstance(response.request_time, datetime)
    assert response.input_payload is not None


@pytest.mark.integ
def test_anthropic_messages_streaming(
    aws_credentials,
    bedrock_anthropic_mantle_region,
    bedrock_anthropic_mantle_test_model,
):
    """
    Test AnthropicMessagesStream endpoint with Bedrock Mantle (streaming).

    Validates that the endpoint can:
    - Initialize with Bedrock Mantle provider and AWS credentials
    - Invoke the streaming Anthropic Messages API
    - Return an InvocationResponse with text, TTFT, TTLT, and token counts
    - Verify TTLT > TTFT
    - Complete without errors

    Args:
        aws_credentials: Boto3 session with valid AWS credentials.
        bedrock_anthropic_mantle_region: AWS region for Bedrock Mantle.
        bedrock_anthropic_mantle_test_model: Anthropic model ID for Bedrock Mantle.

    Estimated Cost: ~$0.001 per run
    """
    endpoint = AnthropicMessagesStream(
        model_id=bedrock_anthropic_mantle_test_model,
        provider="bedrock-mantle",
        aws_region=bedrock_anthropic_mantle_region,
    )

    payload = AnthropicMessagesEndpoint.create_payload(
        user_message="Hello, this is a test. Please respond with a brief greeting.",
        max_tokens=100,
    )

    response = endpoint.invoke(payload)

    # No errors
    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )

    # Response text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert isinstance(response.response_text, str)

    # Token counts
    assert response.num_tokens_input is not None, "Input token count should be present"
    assert response.num_tokens_input > 0, "Input token count should be positive"
    assert response.num_tokens_output is not None, (
        "Output token count should be present"
    )
    assert response.num_tokens_output > 0, "Output token count should be positive"

    # TTFT
    assert response.time_to_first_token is not None, "TTFT should be present"
    assert response.time_to_first_token > 0, "TTFT should be positive"

    # TTLT
    assert response.time_to_last_token is not None, "TTLT should be present"
    assert response.time_to_last_token > 0, "TTLT should be positive"

    # TTLT > TTFT
    assert response.time_to_last_token > response.time_to_first_token, (
        "TTLT should be greater than TTFT"
    )

    # Response ID
    assert response.id is not None, "Response should have an ID"
    assert response.id.startswith("msg_"), (
        f"Response ID should be Anthropic format (msg_...), got: {response.id}"
    )

    # Metadata back-fill
    assert isinstance(response.request_time, datetime)


@pytest.mark.integ
def test_anthropic_messages_create_payload_roundtrip(
    aws_credentials,
    bedrock_anthropic_mantle_region,
    bedrock_anthropic_mantle_test_model,
):
    """
    Test that create_payload output works end-to-end with the endpoint.

    Validates the full flow: create_payload → invoke → valid response,
    including extra kwargs like system prompt and temperature.

    Args:
        aws_credentials: Boto3 session with valid AWS credentials.
        bedrock_anthropic_mantle_region: AWS region for Bedrock Mantle.
        bedrock_anthropic_mantle_test_model: Anthropic model ID for Bedrock Mantle.

    Estimated Cost: ~$0.001 per run
    """
    endpoint = AnthropicMessages(
        model_id=bedrock_anthropic_mantle_test_model,
        provider="bedrock-mantle",
        aws_region=bedrock_anthropic_mantle_region,
    )

    payload = AnthropicMessagesEndpoint.create_payload(
        user_message="What is 2 + 2? Answer with just the number.",
        max_tokens=10,
        system="You are a calculator. Only output numbers.",
    )

    response = endpoint.invoke(payload)

    assert response.error is None, f"Unexpected error: {response.error}"
    assert response.response_text is not None
    assert "4" in response.response_text, (
        f"Expected '4' in response, got: {response.response_text}"
    )


@pytest.mark.integ
def test_anthropic_messages_thinking_token_accounting(
    aws_credentials,
    bedrock_anthropic_mantle_region,
    bedrock_anthropic_mantle_test_model,
):
    """`usage.output_tokens_details.thinking_tokens` is read into num_tokens_output_reasoning.

    This is the assertion that catches the field being renamed, dropped, or surfaced in a shape
    our extraction doesn't handle -- note the anthropic SDK returns it as a plain dict on
    versions that don't model it, and as an object on versions that do.

    Also verifies the downstream consequence: with a thinking-token count available, TPOT is
    derivable even in `display: "omitted"` mode where TTFT is unmeasurable.

    Estimated Cost: ~$0.001 per run
    """
    endpoint = AnthropicMessagesStream(
        model_id=bedrock_anthropic_mantle_test_model,
        provider="bedrock-mantle",
        aws_region=bedrock_anthropic_mantle_region,
    )

    payload = AnthropicMessagesEndpoint.create_payload(
        "What is 15 * 37? Reply with just the number.",
        max_tokens=2048,
        thinking={"type": "adaptive", "display": "omitted"},
    )

    response = endpoint.invoke(payload)

    assert response.error is None, f"Response error: {response.error}"

    assert response.num_tokens_output is not None, (
        "Output token count should not be None"
    )
    assert response.num_tokens_output_reasoning is not None, (
        "Anthropic reports usage.output_tokens_details.thinking_tokens, so "
        "num_tokens_output_reasoning should be populated. If this fails, check whether the "
        "field has been renamed or is arriving in an unexpected shape."
    )
    assert response.num_tokens_output_reasoning > 0, (
        "Expected a non-zero thinking-token count with thinking enabled, got "
        f"{response.num_tokens_output_reasoning}"
    )
    assert response.num_tokens_output_reasoning <= response.num_tokens_output, (
        f"Thinking tokens ({response.num_tokens_output_reasoning}) should be <= total output "
        f"tokens ({response.num_tokens_output}) -- the total is inclusive"
    )

    # display="omitted" suppresses thinking deltas, so first-token time is unmeasurable...
    assert response.time_to_first_token is None, (
        "TTFT should be None in omitted mode, where no thinking deltas are streamed"
    )
    assert response.time_to_first_content_token is not None, (
        "Content TTFT should still be measured in omitted mode"
    )
    # ...but the signature time is preserved for reference
    assert "anthropic_time_to_thinking_signature" in response.annotations, (
        "The signature arrival time should be recorded as an annotation"
    )

    # ...and the ingredients for the visible-token TPOT fallback are all present. The fallback
    # arithmetic itself is covered by the unit tests; what only a live call can confirm is that
    # the provider supplies the inputs it needs.
    assert response.time_to_last_token is not None
    assert response.num_tokens_output - response.num_tokens_output_reasoning >= 0, (
        "Visible token count should not be negative"
    )
