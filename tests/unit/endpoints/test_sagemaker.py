# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from io import BytesIO
from unittest.mock import patch

import pytest
import requests
from moto import mock_aws

from llmeter.endpoints.sagemaker import (
    InvocationResponse,
    SageMakerBase,
    SageMakerEndpoint,
    SageMakerStreamEndpoint,
    TokenIterator,
)


class ConcreteClass(SageMakerBase):
    @SageMakerBase.llmeter_invoke
    def invoke(self, payload):
        return None

    def process_raw_response(self, raw_response, start_t, response):
        response.response_text = "test response"


@pytest.fixture
def sagemaker_base():
    with mock_aws():
        return ConcreteClass(endpoint_name="test-endpoint", model_id="test-model")


@pytest.fixture
def sagemaker_endpoint():
    with mock_aws():
        return SageMakerEndpoint(endpoint_name="test-endpoint", model_id="test-model")


@pytest.fixture
def sagemaker_stream_endpoint():
    with mock_aws():
        return SageMakerStreamEndpoint(
            endpoint_name="test-endpoint", model_id="test-model"
        )


def test_sagemaker_base_init(sagemaker_base: ConcreteClass):
    assert sagemaker_base.endpoint_name == "test-endpoint"
    assert sagemaker_base.model_id == "test-model"
    assert sagemaker_base.provider == "sagemaker"
    assert sagemaker_base.generated_text_jmespath == "generated_text"
    assert sagemaker_base.input_text_jmespath == "inputs"
    assert sagemaker_base.token_count_jmespath == "details.generated_tokens"


def test_sagemaker_base_parse_input(sagemaker_base: ConcreteClass):
    payload = {"inputs": "Test input"}
    assert sagemaker_base._parse_input(payload) == "Test input"


def test_sagemaker_sync_create_payload():
    payload = SageMakerEndpoint.create_payload("Test input", max_tokens=100)
    assert payload == {
        "inputs": "Test input",
        "parameters": {
            "max_new_tokens": 100,
            "details": True,
        },
    }


def test_sagemaker_stream_create_payload():
    payload = SageMakerStreamEndpoint.create_payload("Test input", max_tokens=100)
    assert payload == {
        "inputs": "Test input",
        "parameters": {
            "max_new_tokens": 100,
            "details": True,
        },
        "stream": True,
    }


# @patch("boto3.client")
def test_sagemaker_endpoint_invoke(sagemaker_endpoint: SageMakerEndpoint):
    expected_results = {
        "region": sagemaker_endpoint.region,
        "results": [
            {
                "Body": f"""{
                    json.dumps(
                        {
                            "generated_text": "Test output",
                            "details": {"generated_tokens": 10},
                        }
                    )
                }""",
                "ContentType": "application/json",
                # "InvokedProductionVariant": "prod",
                # "CustomAttributes": "my_attr",
            },
        ],
    }
    with mock_aws():
        requests.post(
            "https://motoapi.amazonaws.com/moto-api/static/sagemaker/endpoint-results",
            json=expected_results,
        )

        payload = {"inputs": "Test input"}
        response = sagemaker_endpoint.invoke(payload)

    assert isinstance(response, InvocationResponse)
    assert response.response_text == "Test output"
    assert response.num_tokens_output == 10
    assert isinstance(response.id, str)
    assert len(response.id) > 0


@patch("time.perf_counter")
def test_sagemaker_endpoint_time_to_last_token_is_elapsed(
    mock_perf_counter, sagemaker_endpoint: SageMakerEndpoint
):
    """TTLT must be a duration relative to `start_t`, not a raw perf_counter reading.

    Regression test: `process_raw_response` previously assigned `time.perf_counter()`
    without subtracting `start_t`, so `SageMakerEndpoint` reported an absolute clock
    value (time since boot) instead of the request latency.
    """
    start_t = 100.0
    mock_perf_counter.return_value = 100.25

    raw_response = {
        "Body": BytesIO(
            json.dumps(
                {
                    "generated_text": "Test output",
                    "details": {"generated_tokens": 10},
                }
            ).encode("utf-8")
        )
    }

    response = InvocationResponse(response_text=None)
    sagemaker_endpoint.process_raw_response(raw_response, start_t, response)

    assert response.time_to_last_token is not None
    assert abs(response.time_to_last_token - 0.25) < 1e-5


# @patch("boto3.client")
# def test_sagemaker_stream_endpoint_invoke(mock_boto3_client, sagemaker_stream_endpoint: SageMakerStreamEndpoint):
#     mock_client = Mock()
#     mock_boto3_client.return_value = mock_client

#     class MockStream:
#         def __init__(self):
#             self.content = [
#                 b'data: {"token": {"text": "Hello"}}\n',
#                 b'data: {"token": {"text": " World"}}\n',
#                 b'data: {"token": {"text": "!"}}\n',
#             ]
#             self.index = 0

#         def __iter__(self):
#             return self

#         def __next__(self):
#             if self.index < len(self.content):
#                 result = self.content[self.index]
#                 self.index += 1
#                 return result
#             raise StopIteration

#     mock_response = {"Body": MockStream()}
#     mock_client.invoke_endpoint_with_response_stream.return_value = mock_response

#     payload = {"inputs": "Test input"}
#     response = sagemaker_stream_endpoint.invoke(payload)

#     assert isinstance(response, InvocationResponse)
#     assert response.response_text == "Hello World!"
#     assert response.num_tokens_output == 3
#     assert isinstance(response.id, str)
#     assert UUID(response.id, version=4)


def test_sagemaker_endpoint_error_handling(sagemaker_endpoint: SageMakerEndpoint):
    with patch.object(sagemaker_endpoint, "_sagemaker_runtime") as mock_runtime:
        mock_runtime.invoke_endpoint.side_effect = Exception("Test error")

        payload = {"inputs": "Test input"}
        response = sagemaker_endpoint.invoke(payload)

        assert isinstance(response, InvocationResponse)
        assert response.error == "Test error"


def test_sagemaker_stream_endpoint_error_handling(
    sagemaker_stream_endpoint: SageMakerStreamEndpoint,
):
    with patch.object(sagemaker_stream_endpoint, "_sagemaker_runtime") as mock_runtime:
        mock_runtime.invoke_endpoint_with_response_stream.side_effect = Exception(
            "Test error"
        )

        payload = {"inputs": "Test input", "stream": True}
        response = sagemaker_stream_endpoint.invoke(payload)

        assert isinstance(response, InvocationResponse)
        assert response.error == "Test error"


@pytest.fixture
def mock_stream():
    stream = [
        {"token": {"text": k}}
        for k in "sample text to validate the TokenIterator".split()
    ]
    stream[-1] = {**stream[-1], "details": {"generated_tokens": len(stream)}}
    return [
        {"PayloadPart": {"Bytes": json.dumps(s).encode("utf-8") + b"\n"}}
        for s in stream
    ]


def test_token_iterator_next(mock_stream):
    iterator = TokenIterator(mock_stream)
    assert next(iterator) == "sample"
    assert next(iterator) == "text"
    assert next(iterator) == "to"
    # The last item contains details, not a token

    # with pytest.raises(StopIteration):
    [_ for _ in iterator]

    assert iterator.details == {"generated_tokens": 6}


def test_token_iterator_error_handling():
    error_stream = [{"PayloadPart": {"Bytes": b'data: {"error": "Test error"}\n'}}]
    iterator = TokenIterator(error_stream)

    with pytest.raises(Exception) as exc_info:
        next(iterator)

    assert str(exc_info.value) == "Test error"


def _payload_stream(lines: list[str]):
    """Wrap raw SSE lines as SageMaker `PayloadPart` events."""
    return [{"PayloadPart": {"Bytes": (ln + "\n").encode("utf-8")}} for ln in lines]


def test_token_iterator_rejects_chat_completions_schema_with_actionable_error():
    """A non-TGI chunk must fail with an explanation, not a bare `KeyError: 'token'`.

    SageMaker only transports bytes -- the schema is the container's. LMI and the `/openai/v1`
    endpoint path both serve OpenAI Chat Completions, which this parser cannot read.
    """
    chunk = json.dumps({"choices": [{"delta": {"content": "Hi"}}]})
    iterator = TokenIterator(_payload_stream([f"data:{chunk}"]))

    with pytest.raises(ValueError) as exc_info:
        next(iterator)

    message = str(exc_info.value)
    assert "TGI schema" in message
    assert "choices" in message, "Error should report the keys actually present"
    assert "OpenAICompletionStreamEndpoint" in message, (
        "Error should point at the connector that can read this schema"
    )


def test_token_iterator_reasoning_chunk_also_reports_clearly():
    """A reasoning-carrying Chat Completions chunk fails the same diagnosable way."""
    chunk = json.dumps({"choices": [{"delta": {"reasoning_content": "thinking"}}]})
    iterator = TokenIterator(_payload_stream([f"data:{chunk}"]))

    with pytest.raises(ValueError, match="TGI schema"):
        next(iterator)


def test_token_iterator_still_parses_tgi_schema():
    """The supported schema must keep working unchanged."""
    lines = [
        'data:{"token": {"text": "Hello"}}',
        'data:{"token": {"text": " world"}, "details": {"generated_tokens": 2}}',
    ]
    iterator = TokenIterator(_payload_stream(lines))

    assert list(iterator) == ["Hello", " world"]
    assert iterator.details == {"generated_tokens": 2}
