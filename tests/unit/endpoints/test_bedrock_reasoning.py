# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for reasoning token parsing in BedrockConverseStream.

Covers:
- ``time_to_first_token`` is set by the first delta of any kind, including
  ``reasoningContent`` deltas.
- ``time_to_first_content_token`` is set only by the first visible ``text`` delta.
- Reasoning deltas never contribute to ``response_text``.
- The retired ``ttft_visible_tokens_only`` argument warns and is ignored.
"""

import time
from unittest.mock import patch

import pytest

from llmeter.endpoints.base import InvocationResponse
from llmeter.endpoints.bedrock import BedrockConverseStream


def _make_draft_response() -> InvocationResponse:
    return InvocationResponse(response_text=None, id=None)


def _stream_response(stream_chunks: list[dict]) -> dict:
    """Wrap stream chunks in the Bedrock ConverseStream response envelope."""
    return {
        "stream": stream_chunks,
        "ResponseMetadata": {"RequestId": "req-123", "RetryAttempts": 0},
    }


def _reasoning_delta(text: str = "thinking...") -> dict:
    return {"contentBlockDelta": {"delta": {"reasoningContent": {"text": text}}}}


def _text_delta(text: str) -> dict:
    return {"contentBlockDelta": {"delta": {"text": text}}}


_USAGE_CHUNK = {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}}


# ---------------------------------------------------------------------------
# Tests: the two first-token metrics
# ---------------------------------------------------------------------------


class TestFirstTokenMetrics:
    """`time_to_first_token` covers any token; `time_to_first_content_token` only visible ones."""

    @patch("time.perf_counter")
    def test_reasoning_sets_ttft_and_text_sets_content_ttft(self, mock_perf_counter):
        """The two metrics must resolve to the reasoning and text deltas respectively."""
        # Calls: reasoning_delta, text_delta, contentBlockStop, metadata
        mock_perf_counter.side_effect = [100.2, 100.5, 100.6, 100.7]

        endpoint = BedrockConverseStream(model_id="test-model")
        raw = _stream_response(
            [
                _reasoning_delta(),
                _text_delta("Answer"),
                {"contentBlockStop": {}},
                _USAGE_CHUNK,
            ]
        )

        response = _make_draft_response()
        endpoint.process_raw_response(raw, 100.0, response)

        assert response.time_to_first_token == pytest.approx(0.2)
        assert response.time_to_first_content_token == pytest.approx(0.5)
        assert response.time_to_last_token == pytest.approx(0.6)

    def test_multiple_reasoning_deltas_do_not_overwrite_ttft(self):
        """Only the *first* reasoning delta sets TTFT."""
        endpoint = BedrockConverseStream(model_id="test-model")
        raw = _stream_response(
            [
                _reasoning_delta("step 1"),
                _reasoning_delta("step 2"),
                _text_delta("Answer"),
                {"contentBlockStop": {}},
                _USAGE_CHUNK,
            ]
        )

        response = _make_draft_response()
        with patch("time.perf_counter") as clock:
            clock.side_effect = [100.2, 100.3, 100.5, 100.6, 100.7]
            endpoint.process_raw_response(raw, 100.0, response)

        assert response.time_to_first_token == pytest.approx(0.2)
        assert response.time_to_first_content_token == pytest.approx(0.5)

    def test_multiple_text_deltas_do_not_overwrite_content_ttft(self):
        """Only the *first* text delta sets the content TTFT."""
        endpoint = BedrockConverseStream(model_id="test-model")
        raw = _stream_response(
            [
                _text_delta("Hello"),
                _text_delta(" world"),
                {"contentBlockStop": {}},
                _USAGE_CHUNK,
            ]
        )

        response = _make_draft_response()
        with patch("time.perf_counter") as clock:
            clock.side_effect = [100.4, 100.5, 100.6, 100.7]
            endpoint.process_raw_response(raw, 100.0, response)

        assert response.time_to_first_content_token == pytest.approx(0.4)
        assert response.response_text == "Hello world"

    def test_metrics_equal_without_reasoning(self):
        """With no reasoning content the two metrics describe the same token."""
        endpoint = BedrockConverseStream(model_id="test-model")
        raw = _stream_response(
            [_text_delta("Hello"), {"contentBlockStop": {}}, _USAGE_CHUNK]
        )

        response = _make_draft_response()
        endpoint.process_raw_response(raw, time.perf_counter(), response)

        assert response.time_to_first_token is not None
        assert response.time_to_first_token == response.time_to_first_content_token

    def test_empty_text_delta_does_not_set_timings(self):
        """An empty text delta is not a token."""
        endpoint = BedrockConverseStream(model_id="test-model")
        raw = _stream_response(
            [_text_delta(""), _text_delta("Hi"), {"contentBlockStop": {}}, _USAGE_CHUNK]
        )

        response = _make_draft_response()
        with patch("time.perf_counter") as clock:
            clock.side_effect = [100.2, 100.5, 100.6, 100.7]
            endpoint.process_raw_response(raw, 100.0, response)

        assert response.time_to_first_token == pytest.approx(0.5)
        assert response.time_to_first_content_token == pytest.approx(0.5)

    def test_only_reasoning_no_text(self):
        """Reasoning-only stream: TTFT is set, but there is no content token to time."""
        endpoint = BedrockConverseStream(model_id="test-model")
        raw = _stream_response(
            [
                _reasoning_delta(),
                {"contentBlockStop": {}},
                {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 0}}},
            ]
        )

        response = _make_draft_response()
        endpoint.process_raw_response(raw, time.perf_counter(), response)

        assert response.time_to_first_token is not None
        assert response.time_to_first_content_token is None
        assert response.response_text is None


# ---------------------------------------------------------------------------
# Tests: response_text must never include reasoning
# ---------------------------------------------------------------------------


class TestResponseText:
    def test_reasoning_deltas_excluded_from_response_text(self):
        endpoint = BedrockConverseStream(model_id="test-model")
        raw = _stream_response(
            [
                _reasoning_delta("internal thought"),
                _text_delta("Visible"),
                _text_delta(" answer"),
                {"contentBlockStop": {}},
                _USAGE_CHUNK,
            ]
        )

        response = _make_draft_response()
        endpoint.process_raw_response(raw, time.perf_counter(), response)

        assert response.response_text == "Visible answer"
        assert "internal thought" not in (response.response_text or "")


# ---------------------------------------------------------------------------
# Tests: deprecated constructor argument
# ---------------------------------------------------------------------------


class TestDeprecatedTtftFlag:
    @pytest.mark.parametrize("value", [True, False])
    def test_passing_flag_warns(self, value):
        with pytest.warns(DeprecationWarning, match="ttft_visible_tokens_only"):
            BedrockConverseStream(model_id="test-model", ttft_visible_tokens_only=value)

    def test_omitting_flag_does_not_warn(self, recwarn):
        BedrockConverseStream(model_id="test-model")
        assert [w for w in recwarn if w.category is DeprecationWarning] == []

    def test_flag_does_not_change_behaviour(self):
        """Both metrics are recorded regardless of the retired flag's value."""
        raw_chunks = [
            _reasoning_delta(),
            _text_delta("Answer"),
            {"contentBlockStop": {}},
            _USAGE_CHUNK,
        ]

        results = []
        for value in (True, False):
            with pytest.warns(DeprecationWarning):
                endpoint = BedrockConverseStream(
                    model_id="test-model", ttft_visible_tokens_only=value
                )
            response = _make_draft_response()
            with patch("time.perf_counter") as clock:
                clock.side_effect = [100.2, 100.5, 100.6, 100.7]
                endpoint.process_raw_response(
                    _stream_response(raw_chunks), 100.0, response
                )
            results.append(
                (response.time_to_first_token, response.time_to_first_content_token)
            )

        assert results[0] == results[1]
        assert results[0] == (pytest.approx(0.2), pytest.approx(0.5))
