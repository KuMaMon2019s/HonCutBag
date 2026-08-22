from __future__ import annotations

import hashlib

import pytest

from runtime.execution_errors import SubmissionUncertainError
from runtime.video_provider import (
    ProviderErrorKind,
    VideoGenerationRequest,
    VideoJobState,
    VideoJobStatus,
    VideoProvider,
    VideoProviderCapabilities,
    VideoSubmission,
    classify_provider_error,
)


class FakeProvider:
    provider_id = "fake"
    capabilities = VideoProviderCapabilities(
        provider_id="fake",
        generation_modes=frozenset({"txt2vid", "img2vid"}),
        supports_cancel=True,
    )

    def submit(self, request: VideoGenerationRequest) -> VideoSubmission:
        assert request.resource_id == "S01"
        return VideoSubmission("job-1", "https://provider.example/jobs")

    def status(self, provider_job_id: str) -> VideoJobStatus:
        return VideoJobStatus(
            VideoJobState.SUCCEEDED,
            output_uri=f"https://provider.example/{provider_job_id}.mp4",
        )

    def cancel(self, provider_job_id: str) -> VideoJobStatus:
        return VideoJobStatus(VideoJobState.CANCELLED)


def test_request_is_canonical_and_detached_from_mutable_input():
    payload = {"prompt": "雨夜", "params": {"duration": 5}}
    fingerprint = hashlib.sha256(b"semantic-input").hexdigest()
    request = VideoGenerationRequest.from_payload(
        resource_id="S01",
        input_fingerprint=fingerprint,
        payload=payload,
    )
    payload["params"]["duration"] = 99

    assert request.payload == {"params": {"duration": 5}, "prompt": "雨夜"}
    assert request.payload_json == '{"params":{"duration":5},"prompt":"雨夜"}'


def test_provider_protocol_normalizes_submit_status_and_cancel():
    provider = FakeProvider()
    assert isinstance(provider, VideoProvider)
    request = VideoGenerationRequest.from_payload(
        resource_id="S01",
        input_fingerprint="0" * 64,
        payload={"prompt": "test"},
    )

    submission = provider.submit(request)
    status = provider.status(submission.provider_job_id)
    cancelled = provider.cancel(submission.provider_job_id)

    assert provider.capabilities.supports_cancel is True
    assert status.state is VideoJobState.SUCCEEDED and status.state.terminal
    assert cancelled.state is VideoJobState.CANCELLED and cancelled.state.terminal


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (SubmissionUncertainError("accepted, id lost"), ProviderErrorKind.SUBMISSION_UNCERTAIN),
        (RuntimeError("Seedance API 429: too many requests"), ProviderErrorKind.RATE_LIMIT),
        (RuntimeError("QuotaExceeded"), ProviderErrorKind.QUOTA),
        (RuntimeError("PrivacyInformation"), ProviderErrorKind.MODERATION),
        (TimeoutError("poll timed out"), ProviderErrorKind.TIMEOUT),
        (RuntimeError("HTTP 503 temporarily unavailable"), ProviderErrorKind.TRANSIENT),
    ],
)
def test_provider_error_classification(error, expected):
    assert classify_provider_error(error) is expected


def test_status_contract_fails_closed_for_missing_terminal_details():
    with pytest.raises(ValueError, match="output_uri"):
        VideoJobStatus(VideoJobState.SUCCEEDED)
    with pytest.raises(ValueError, match="error_message"):
        VideoJobStatus(VideoJobState.FAILED)
