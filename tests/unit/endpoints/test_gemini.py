# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from unittest.mock import Mock, patch

import pytest

from llmeter.endpoints.base import InvocationResponse
from llmeter.endpoints.gemini import (
    GeminiEndpoint,
    GeminiEndpointBase,
    GeminiStreamEndpoint,
)

_PATCH_GENAI = "llmeter.endpoints.gemini.genai"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(
    text="Hello, world!",
    prompt_tokens=10,
    output_tokens=5,
    cached_tokens=None,
):
    """Build a mock non-streaming GenerateContentResponse."""
    # Create part with text
    part = Mock()
    part.text = text

    # Create content with parts
    content = Mock()
    content.parts = [part]

    # Create candidate with content
    candidate = Mock()
    candidate.content = content

    # Create usage metadata
    usage = Mock()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = output_tokens
    usage.cached_content_token_count = cached_tokens

    # Create response
    response = Mock()
    response.candidates = [candidate]
    response.usage_metadata = usage

    return response


def _make_stream_chunks(
    text_chunks=None,
    prompt_tokens=10,
    output_tokens=5,
    cached_tokens=None,
):
    """Build a list of mock streaming response chunks."""
    if text_chunks is None:
        text_chunks = ["Hello", ", ", "world!"]

    chunks = []

    # Create chunks for each text part
    for i, text in enumerate(text_chunks):
        part = Mock()
        part.text = text

        content = Mock()
        content.parts = [part]

        candidate = Mock()
        candidate.content = content

        chunk = Mock()
        chunk.candidates = [candidate]

        # Add usage metadata only to final chunk
        if i == len(text_chunks) - 1:
            usage = Mock()
            usage.prompt_token_count = prompt_tokens
            usage.candidates_token_count = output_tokens
            usage.cached_content_token_count = cached_tokens
            chunk.usage_metadata = usage
        else:
            chunk.usage_metadata = None

        chunks.append(chunk)

    return chunks


def _make_draft_response() -> InvocationResponse:
    """Create a draft InvocationResponse like llmeter_invoke does."""
    return InvocationResponse(response_text=None)


@pytest.fixture
def mock_genai():
    """Mock the google.generativeai module."""
    with patch(_PATCH_GENAI) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Tests: create_payload
# ---------------------------------------------------------------------------


class TestCreatePayload:
    def test_basic_payload(self):
        """Test create_payload with basic arguments."""
        payload = GeminiEndpointBase.create_payload("Hello!")
        assert payload == {
            "contents": "Hello!",
            "generation_config": {"max_output_tokens": 256},
        }

    def test_custom_max_tokens(self):
        """Test create_payload with custom max_tokens."""
        payload = GeminiEndpointBase.create_payload("Hi", max_tokens=1024)
        assert payload["generation_config"]["max_output_tokens"] == 1024

    def test_extra_kwargs(self):
        """Test create_payload with additional kwargs."""
        payload = GeminiEndpointBase.create_payload(
            "Hi", max_tokens=512, temperature=0.7, top_p=0.9
        )
        assert payload["generation_config"]["max_output_tokens"] == 512
        assert payload["generation_config"]["temperature"] == 0.7
        assert payload["generation_config"]["top_p"] == 0.9

    def test_invalid_max_tokens(self):
        """Test create_payload with invalid max_tokens."""
        with pytest.raises(ValueError, match="positive integer"):
            GeminiEndpointBase.create_payload("Hi", max_tokens=-1)

    def test_zero_max_tokens(self):
        """Test create_payload with zero max_tokens."""
        with pytest.raises(ValueError, match="positive integer"):
            GeminiEndpointBase.create_payload("Hi", max_tokens=0)

    def test_non_string_message(self):
        """Test create_payload with non-string message."""
        with pytest.raises(TypeError, match="must be a str"):
            GeminiEndpointBase.create_payload(123)


# ---------------------------------------------------------------------------
# Tests: _parse_payload
# ---------------------------------------------------------------------------


class TestParsePayload:
    def test_parse_string_content(self, mock_genai):
        """Test _parse_payload with string content."""
        endpoint = GeminiEndpoint(model_id="test-model", api_key="test-key")
        payload = {"contents": "Hello"}
        assert endpoint._parse_payload(payload) == "Hello"

    def test_parse_list_content(self, mock_genai):
        """Test _parse_payload with list of dict content."""
        endpoint = GeminiEndpoint(model_id="test-model", api_key="test-key")
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": "Hello"},
                        {"text": "World"},
                    ]
                }
            ]
        }
        assert endpoint._parse_payload(payload) == "Hello\nWorld"

    def test_parse_empty_contents(self, mock_genai):
        """Test _parse_payload with empty contents."""
        endpoint = GeminiEndpoint(model_id="test-model", api_key="test-key")
        assert endpoint._parse_payload({"contents": []}) == ""

    def test_parse_no_contents_key(self, mock_genai):
        """Test _parse_payload with missing contents key."""
        endpoint = GeminiEndpoint(model_id="test-model", api_key="test-key")
        assert endpoint._parse_payload({}) == ""

    def test_parse_mixed_content_types(self, mock_genai):
        """Test _parse_payload with mixed content types."""
        endpoint = GeminiEndpoint(model_id="test-model", api_key="test-key")
        payload = {
            "contents": [
                {"parts": [{"text": "Describe this:"}, {"inline_data": {}}]},
            ]
        }
        assert endpoint._parse_payload(payload) == "Describe this:"


# ---------------------------------------------------------------------------
# Tests: GeminiEndpoint (non-streaming)
# ---------------------------------------------------------------------------


class TestGeminiEndpoint:
    def test_initialization(self, mock_genai):
        """Test GeminiEndpoint initialization."""
        endpoint = GeminiEndpoint(
            model_id="gemini-2.0-flash-exp",
            endpoint_name="test_gemini",
            api_key="test_key",
        )

        assert endpoint.model_id == "gemini-2.0-flash-exp"
        assert endpoint.endpoint_name == "test_gemini"
        assert endpoint.provider == "google"
        mock_genai.configure.assert_called_once_with(api_key="test_key")

    def test_initialization_with_custom_provider(self, mock_genai):
        """Test GeminiEndpoint initialization with custom provider."""
        endpoint = GeminiEndpoint(
            model_id="gemini-2.0-flash-exp",
            provider="custom_google",
            api_key="test_key",
        )

        assert endpoint.provider == "custom_google"

    def test_initialization_with_generation_config(self, mock_genai):
        """Test GeminiEndpoint initialization with generation config."""
        endpoint = GeminiEndpoint(
            model_id="gemini-2.0-flash-exp",
            api_key="test_key",
            temperature=0.7,
            top_p=0.9,
        )

        assert endpoint._generation_config["temperature"] == 0.7
        assert endpoint._generation_config["top_p"] == 0.9

    def test_process_raw_response(self, mock_genai):
        """Test process_raw_response method."""
        endpoint = GeminiEndpoint(model_id="test-model", api_key="test-key")
        mock_response = _make_mock_response()
        response = _make_draft_response()

        endpoint.process_raw_response(mock_response, time.perf_counter(), response)

        assert response.response_text == "Hello, world!"
        assert response.num_tokens_input == 10
        assert response.num_tokens_output == 5
        assert response.time_to_last_token is not None

    def test_process_raw_response_with_cache(self, mock_genai):
        """Test process_raw_response with cached tokens."""
        endpoint = GeminiEndpoint(model_id="test-model", api_key="test-key")
        mock_response = _make_mock_response(cached_tokens=3)
        response = _make_draft_response()

        endpoint.process_raw_response(mock_response, time.perf_counter(), response)

        assert response.num_tokens_input_cached == 3

    def test_process_raw_response_no_usage(self, mock_genai):
        """Test process_raw_response with no usage metadata."""
        endpoint = GeminiEndpoint(model_id="test-model", api_key="test-key")
        mock_response = _make_mock_response()
        mock_response.usage_metadata = None
        response = _make_draft_response()

        endpoint.process_raw_response(mock_response, time.perf_counter(), response)

        assert response.num_tokens_input is None
        assert response.num_tokens_output is None
        assert response.num_tokens_input_cached is None

    def test_process_raw_response_multiple_candidates(self, mock_genai):
        """Test process_raw_response with multiple candidates."""
        endpoint = GeminiEndpoint(model_id="test-model", api_key="test-key")

        # Create first candidate
        part1 = Mock()
        part1.text = "Part 1. "
        content1 = Mock()
        content1.parts = [part1]
        candidate1 = Mock()
        candidate1.content = content1

        # Create second candidate
        part2 = Mock()
        part2.text = "Part 2."
        content2 = Mock()
        content2.parts = [part2]
        candidate2 = Mock()
        candidate2.content = content2

        mock_response = _make_mock_response()
        mock_response.candidates = [candidate1, candidate2]
        response = _make_draft_response()

        endpoint.process_raw_response(mock_response, time.perf_counter(), response)
        assert response.response_text == "Part 1. Part 2."

    def test_invoke_success(self, mock_genai):
        """Test successful invoke call."""
        mock_model = Mock()
        mock_model.generate_content.return_value = _make_mock_response()
        mock_genai.GenerativeModel.return_value = mock_model

        endpoint = GeminiEndpoint(model_id="test-model", api_key="test-key")
        result = endpoint.invoke(
            {
                "contents": "Hello",
                "generation_config": {"max_output_tokens": 256},
            }
        )

        assert isinstance(result, InvocationResponse)
        assert result.response_text == "Hello, world!"
        assert result.error is None

    def test_invoke_api_error(self, mock_genai):
        """Test invoke with API error."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API error")
        mock_genai.GenerativeModel.return_value = mock_model

        endpoint = GeminiEndpoint(model_id="test-model", api_key="test-key")
        result = endpoint.invoke(
            {
                "contents": "Hello",
                "generation_config": {"max_output_tokens": 256},
            }
        )

        assert isinstance(result, InvocationResponse)
        assert result.error is not None
        assert "API error" in result.error

    def test_prepare_payload_merges_config(self, mock_genai):
        """Test prepare_payload merges generation config."""
        endpoint = GeminiEndpoint(
            model_id="test-model", api_key="test-key", temperature=0.5
        )
        payload = {
            "contents": "Hi",
            "generation_config": {"max_output_tokens": 256},
        }

        prepared = endpoint.prepare_payload(payload)

        assert prepared["generation_config"]["temperature"] == 0.5
        assert prepared["generation_config"]["max_output_tokens"] == 256

    def test_invoke_sets_input_prompt(self, mock_genai):
        """Test that invoke sets the input_prompt correctly."""
        mock_model = Mock()
        mock_model.generate_content.return_value = _make_mock_response()
        mock_genai.GenerativeModel.return_value = mock_model

        endpoint = GeminiEndpoint(model_id="test-model", api_key="test-key")
        payload = {
            "contents": "Hello, Gemini!",
            "generation_config": {"max_output_tokens": 256},
        }

        response = endpoint.invoke(payload)

        assert response.input_prompt == "Hello, Gemini!"


# ---------------------------------------------------------------------------
# Tests: GeminiStreamEndpoint (streaming)
# ---------------------------------------------------------------------------


class TestGeminiStreamEndpoint:
    def test_initialization(self, mock_genai):
        """Test GeminiStreamEndpoint initialization."""
        endpoint = GeminiStreamEndpoint(
            model_id="gemini-2.0-flash-exp",
            endpoint_name="test_gemini_stream",
            api_key="test_key",
        )

        assert endpoint.model_id == "gemini-2.0-flash-exp"
        assert endpoint.endpoint_name == "test_gemini_stream"
        assert endpoint.provider == "google"

    def test_process_raw_response_basic(self, mock_genai):
        """Test process_raw_response with basic streaming."""
        endpoint = GeminiStreamEndpoint(model_id="test-model", api_key="test-key")
        chunks = _make_stream_chunks()
        response = _make_draft_response()

        endpoint.process_raw_response(iter(chunks), time.perf_counter(), response)

        assert response.response_text == "Hello, world!"
        assert response.num_tokens_input == 10
        assert response.num_tokens_output == 5
        assert response.time_to_first_token is not None
        assert response.time_to_last_token is not None
        assert response.time_to_first_token <= response.time_to_last_token

    def test_process_raw_response_with_cache(self, mock_genai):
        """Test process_raw_response with cached tokens."""
        endpoint = GeminiStreamEndpoint(model_id="test-model", api_key="test-key")
        chunks = _make_stream_chunks(cached_tokens=7)
        response = _make_draft_response()

        endpoint.process_raw_response(iter(chunks), time.perf_counter(), response)

        assert response.num_tokens_input_cached == 7

    def test_process_raw_response_empty_stream(self, mock_genai):
        """Test process_raw_response with empty stream."""
        endpoint = GeminiStreamEndpoint(model_id="test-model", api_key="test-key")
        response = _make_draft_response()

        endpoint.process_raw_response(iter([]), time.perf_counter(), response)

        assert response.response_text is None

    def test_process_raw_response_timing(self, mock_genai):
        """Test process_raw_response timing measurements."""
        endpoint = GeminiStreamEndpoint(model_id="test-model", api_key="test-key")
        chunks = _make_stream_chunks(text_chunks=["First", " Second"])
        response = _make_draft_response()

        start_t = time.perf_counter()
        endpoint.process_raw_response(iter(chunks), start_t, response)

        assert response.time_to_first_token is not None
        assert response.time_to_last_token is not None
        assert response.time_to_first_token > 0
        assert response.time_to_last_token >= response.time_to_first_token

    def test_invoke_success(self, mock_genai):
        """Test successful streaming invoke call."""
        mock_model = Mock()
        mock_model.generate_content.return_value = iter(_make_stream_chunks(["Hi!"]))
        mock_genai.GenerativeModel.return_value = mock_model

        endpoint = GeminiStreamEndpoint(model_id="test-model", api_key="test-key")
        result = endpoint.invoke(
            {
                "contents": "Hello",
                "generation_config": {"max_output_tokens": 256},
            }
        )

        assert isinstance(result, InvocationResponse)
        assert result.response_text == "Hi!"
        assert result.error is None

    def test_invoke_api_error(self, mock_genai):
        """Test invoke with API error."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("Stream error")
        mock_genai.GenerativeModel.return_value = mock_model

        endpoint = GeminiStreamEndpoint(model_id="test-model", api_key="test-key")
        result = endpoint.invoke(
            {
                "contents": "Hello",
                "generation_config": {"max_output_tokens": 256},
            }
        )

        assert isinstance(result, InvocationResponse)
        assert result.error is not None
        assert "Stream error" in result.error

    def test_process_raw_response_no_text_chunks(self, mock_genai):
        """Test process_raw_response with no text chunks."""
        endpoint = GeminiStreamEndpoint(model_id="test-model", api_key="test-key")
        chunks = _make_stream_chunks(text_chunks=[])
        response = _make_draft_response()

        endpoint.process_raw_response(iter(chunks), time.perf_counter(), response)

        assert response.response_text is None
        assert response.time_to_first_token is None

    def test_invoke_sets_input_prompt(self, mock_genai):
        """Test that invoke sets the input_prompt correctly."""
        mock_model = Mock()
        mock_model.generate_content.return_value = iter(_make_stream_chunks(["Hello!"]))
        mock_genai.GenerativeModel.return_value = mock_model

        endpoint = GeminiStreamEndpoint(model_id="test-model", api_key="test-key")
        payload = {
            "contents": "Hello, Gemini!",
            "generation_config": {"max_output_tokens": 256},
        }

        response = endpoint.invoke(payload)

        assert response.input_prompt == "Hello, Gemini!"


# ---------------------------------------------------------------------------
# Tests: Endpoint integration
# ---------------------------------------------------------------------------


class TestGeminiEndpointIntegration:
    def test_endpoint_inheritance(self, mock_genai):
        """Test that Gemini endpoints properly inherit from base classes."""
        endpoint = GeminiEndpoint(model_id="test-model", api_key="test-key")
        stream_endpoint = GeminiStreamEndpoint(model_id="test-model", api_key="test-key")

        assert isinstance(endpoint, GeminiEndpointBase)
        assert isinstance(stream_endpoint, GeminiEndpointBase)

        # Test that they have the required methods
        assert hasattr(endpoint, "invoke")
        assert hasattr(endpoint, "create_payload")
        assert hasattr(stream_endpoint, "invoke")
        assert hasattr(stream_endpoint, "create_payload")

    def test_endpoint_to_dict(self, mock_genai):
        """Test endpoint serialization to dictionary."""
        endpoint = GeminiEndpoint(
            model_id="gemini-2.0-flash-exp",
            endpoint_name="test_gemini",
            api_key="test_key",
        )

        endpoint_dict = endpoint.to_dict()

        assert endpoint_dict["model_id"] == "gemini-2.0-flash-exp"
        assert endpoint_dict["endpoint_name"] == "test_gemini"
        assert endpoint_dict["provider"] == "google"
        assert endpoint_dict["endpoint_type"] == "GeminiEndpoint"

    def test_create_payload_consistency(self):
        """Test that create_payload works consistently across endpoint types."""
        message = "Test message"

        base_payload = GeminiEndpointBase.create_payload(message)
        endpoint_payload = GeminiEndpoint.create_payload(message)
        stream_payload = GeminiStreamEndpoint.create_payload(message)

        # All should create the same payload structure
        assert base_payload == endpoint_payload == stream_payload
        assert base_payload["contents"] == message

    def test_error_handling_consistency(self, mock_genai):
        """Test that error handling is consistent across endpoint types."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("Test error")
        mock_genai.GenerativeModel.return_value = mock_model

        endpoint = GeminiEndpoint(model_id="test-model", api_key="test-key")
        stream_endpoint = GeminiStreamEndpoint(model_id="test-model", api_key="test-key")

        endpoint_response = endpoint.invoke({"contents": "Test"})
        stream_response = stream_endpoint.invoke({"contents": "Test"})

        # Both should handle errors similarly
        assert "Test error" in endpoint_response.error
        assert "Test error" in stream_response.error
        assert endpoint_response.response_text is None
        assert stream_response.response_text is None


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestGeminiEndpointEdgeCases:
    def test_response_timing_accuracy(self, mock_genai):
        """Test that response timing measurements are accurate."""
        mock_model = Mock()

        def delayed_response(*args, **kwargs):
            time.sleep(0.01)  # 10ms delay
            return _make_mock_response()

        mock_model.generate_content.side_effect = delayed_response
        mock_genai.GenerativeModel.return_value = mock_model

        endpoint = GeminiEndpoint(model_id="test-model", api_key="test-key")
        response = endpoint.invoke({"contents": "Hello"})

        # Verify timing is reasonable (should be at least 10ms)
        if response.time_to_last_token is not None:
            assert response.time_to_last_token >= 0.01
            assert response.time_to_last_token < 1.0  # Should be less than 1 second

    def test_stream_response_with_empty_text(self, mock_genai):
        """Test streaming response with empty text chunks."""
        endpoint = GeminiStreamEndpoint(model_id="test-model", api_key="test-key")

        # Create chunks with empty text
        chunks = _make_stream_chunks(text_chunks=["Hello", "", " world"])

        mock_model = Mock()
        mock_model.generate_content.return_value = iter(chunks)
        mock_genai.GenerativeModel.return_value = mock_model

        response = endpoint.invoke({"contents": "Test"})

        assert response.response_text == "Hello world"

    def test_stream_mid_stream_error(self, mock_genai):
        """Test that errors during stream consumption are caught."""
        endpoint = GeminiStreamEndpoint(model_id="test-model", api_key="test-key")

        def exploding_stream():
            chunk = _make_stream_chunks(["Hello"])[0]
            yield chunk
            raise TimeoutError("Read timed out")

        mock_model = Mock()
        mock_model.generate_content.return_value = exploding_stream()
        mock_genai.GenerativeModel.return_value = mock_model

        response = endpoint.invoke({"contents": "Hi"})

        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        assert "timed out" in response.error.lower()
        assert response.input_payload is not None
