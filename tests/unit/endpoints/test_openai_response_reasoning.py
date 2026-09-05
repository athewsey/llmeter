# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for reasoning token parsing in OpenAIResponseStreamEndpoint.

Covers:
- ``time_to_first_token`` is set by the first delta of any kind, including reasoning
  deltas (``response.reasoning_text.delta``, ``response.reasoning_summary_text.delta``).
- ``time_to_first_content_token`` is set only by the first
  ``response.output_text.delta``.
- Reasoning deltas never contribute to ``response_text``.
- ``num_tokens_output_reasoning`` extraction from ``output_tokens_details``.
- ``num_tokens_input_cached`` extraction from ``input_tokens_details``.
- The retired ``ttft_visible_tokens_only`` argument warns and is ignored.
"""

import time
from unittest.mock import Mock, patch

import pytest

from llmeter.endpoints.base import InvocationResponse
from llmeter.endpoints.openai_response import OpenAIResponseStreamEndpoint


def _make_draft_response() -> InvocationResponse:
    return InvocationResponse(response_text=None)


def _event(event_type: str, **attrs) -> Mock:
    """Build a mock streaming event with the given type and attributes."""
    e = Mock()
    e.type = event_type
    for k, v in attrs.items():
        setattr(e, k, v)
    return e


def _created_event(response_id: str = "resp_123") -> Mock:
    resp = Mock()
    resp.id = response_id
    return _event("response.created", response=resp)


def _text_delta_event(text: str) -> Mock:
    return _event("response.output_text.delta", delta=text)


def _reasoning_text_delta_event() -> Mock:
    return _event("response.reasoning_text.delta", delta="thinking...")


def _reasoning_summary_delta_event() -> Mock:
    return _event("response.reasoning_summary_text.delta", delta="summary...")


def _completed_event(
    input_tokens: int = 10,
    output_tokens: int = 20,
    reasoning_tokens: int | None = None,
    cached_tokens: int | None = None,
) -> Mock:
    usage = Mock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    if cached_tokens is not None:
        details = Mock()
        details.cached_tokens = cached_tokens
        usage.input_tokens_details = details
    else:
        usage.input_tokens_details = None

    if reasoning_tokens is not None:
        output_details = Mock()
        output_details.reasoning_tokens = reasoning_tokens
        usage.output_tokens_details = output_details
    else:
        usage.output_tokens_details = None

    resp = Mock()
    resp.usage = usage
    return _event("response.completed", response=resp)


# ---------------------------------------------------------------------------
# Tests: the two first-token metrics
# ---------------------------------------------------------------------------


class TestFirstTokenMetrics:
    """`time_to_first_token` covers any delta; `time_to_first_content_token` only text."""

    @patch("llmeter.endpoints.openai_response.OpenAI")
    @patch("time.perf_counter")
    def test_reasoning_text_delta_sets_ttft_only(
        self, mock_perf_counter, mock_openai_class
    ):
        """`response.reasoning_text.delta` sets TTFT but not the content TTFT."""
        # Calls: created, reasoning_delta, text_delta, completed
        mock_perf_counter.side_effect = [100.1, 100.2, 100.5, 100.6]

        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")
        events = [
            _created_event(),
            _reasoning_text_delta_event(),
            _text_delta_event("Answer"),
            _completed_event(reasoning_tokens=10),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), 100.0, response)

        assert response.time_to_first_token == pytest.approx(0.2)
        assert response.time_to_first_content_token == pytest.approx(0.5)
        assert response.time_to_last_token == pytest.approx(0.5)
        assert response.response_text == "Answer"

    @patch("llmeter.endpoints.openai_response.OpenAI")
    @patch("time.perf_counter")
    def test_reasoning_summary_delta_sets_ttft_only(
        self, mock_perf_counter, mock_openai_class
    ):
        """`response.reasoning_summary_text.delta` sets TTFT but not the content TTFT."""
        mock_perf_counter.side_effect = [100.1, 100.3, 100.7, 100.8]

        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")
        events = [
            _created_event(),
            _reasoning_summary_delta_event(),
            _text_delta_event("Result"),
            _completed_event(reasoning_tokens=3),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), 100.0, response)

        assert response.time_to_first_token == pytest.approx(0.3)
        assert response.time_to_first_content_token == pytest.approx(0.7)
        assert response.response_text == "Result"

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_ttft_not_overwritten_by_later_deltas(self, mock_openai_class):
        """Only the first reasoning delta sets TTFT; only the first text delta sets content TTFT."""
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")
        events = [
            _created_event(),
            _reasoning_text_delta_event(),
            _reasoning_text_delta_event(),
            _text_delta_event("Fin"),
            _text_delta_event("al"),
            _completed_event(),
        ]

        response = _make_draft_response()
        with patch("time.perf_counter") as clock:
            clock.side_effect = [100.1, 100.2, 100.3, 100.5, 100.6, 100.7]
            endpoint.process_raw_response(iter(events), 100.0, response)

        assert response.time_to_first_token == pytest.approx(0.2)
        assert response.time_to_first_content_token == pytest.approx(0.5)
        assert response.response_text == "Final"

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_metrics_equal_without_reasoning(self, mock_openai_class):
        """With no reasoning deltas the two metrics describe the same token."""
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")
        events = [
            _created_event(),
            _text_delta_event("Hello"),
            _completed_event(),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.time_to_first_token is not None
        assert response.time_to_first_token == response.time_to_first_content_token

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_reasoning_only_leaves_content_ttft_unset(self, mock_openai_class):
        """A reasoning-only stream has no content token to time."""
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")
        events = [
            _created_event(),
            _reasoning_text_delta_event(),
            _completed_event(reasoning_tokens=5),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.time_to_first_token is not None
        assert response.time_to_first_content_token is None
        assert response.response_text is None

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_reasoning_deltas_do_not_contribute_to_response_text(
        self, mock_openai_class
    ):
        """Reasoning deltas must never appear in response_text."""
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")
        events = [
            _created_event(),
            _reasoning_text_delta_event(),
            _reasoning_summary_delta_event(),
            _text_delta_event("Answer"),
            _completed_event(),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.response_text == "Answer"


# ---------------------------------------------------------------------------
# Tests: deprecated constructor argument
# ---------------------------------------------------------------------------


class TestDeprecatedTtftFlag:
    @patch("llmeter.endpoints.openai_response.OpenAI")
    @pytest.mark.parametrize("value", [True, False])
    def test_passing_flag_warns(self, mock_openai_class, value):
        with pytest.warns(DeprecationWarning, match="ttft_visible_tokens_only"):
            OpenAIResponseStreamEndpoint(
                model_id="test-model", ttft_visible_tokens_only=value
            )

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_omitting_flag_does_not_warn(self, mock_openai_class, recwarn):
        OpenAIResponseStreamEndpoint(model_id="test-model")
        assert [w for w in recwarn if w.category is DeprecationWarning] == []


# ---------------------------------------------------------------------------
# Tests: num_tokens_output_reasoning extraction
# ---------------------------------------------------------------------------


class TestReasoningTokenCount:
    """Verify num_tokens_output_reasoning is extracted from output_tokens_details."""

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_reasoning_tokens_extracted(self, mock_openai_class):
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")

        events = [
            _created_event(),
            _text_delta_event("Hi"),
            _completed_event(
                input_tokens=15,
                output_tokens=25,
                reasoning_tokens=12,
            ),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.num_tokens_output_reasoning == 12
        assert response.num_tokens_input == 15
        assert response.num_tokens_output == 25

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_reasoning_tokens_none_when_not_present(self, mock_openai_class):
        """When output_tokens_details is absent, reasoning count stays None."""
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")

        events = [
            _created_event(),
            _text_delta_event("Hi"),
            _completed_event(
                input_tokens=10,
                output_tokens=20,
                reasoning_tokens=None,
            ),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.num_tokens_output_reasoning is None
        assert response.num_tokens_output == 20

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_cached_tokens_extracted(self, mock_openai_class):
        """Verify input_tokens_details.cached_tokens is captured."""
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")

        events = [
            _created_event(),
            _text_delta_event("Hi"),
            _completed_event(
                input_tokens=100,
                output_tokens=20,
                cached_tokens=80,
            ),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.num_tokens_input_cached == 80
        assert response.num_tokens_input == 100

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_reasoning_and_cached_tokens_together(self, mock_openai_class):
        """Both reasoning and cached token counts extracted in the same response."""
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")

        events = [
            _created_event(),
            _text_delta_event("Hi"),
            _completed_event(
                input_tokens=100,
                output_tokens=50,
                reasoning_tokens=30,
                cached_tokens=60,
            ),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.num_tokens_output_reasoning == 30
        assert response.num_tokens_input_cached == 60
        assert response.num_tokens_input == 100
        assert response.num_tokens_output == 50
