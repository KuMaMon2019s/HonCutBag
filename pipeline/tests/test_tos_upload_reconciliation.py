from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx2 as httpx
import pytest

from clients import tos_uploader
from runtime.provider_policy import TOSUploadExecutionPolicy
from runtime.tos_uploads import TOS_UPLOAD_LEDGER_NAME, tos_upload_execution_scope
from utils.provider_request_guard import (
    MediaUploadTimeouts,
    effective_media_upload_timeouts,
)


def _configure_tos(monkeypatch) -> None:
    monkeypatch.setattr(
        tos_uploader,
        "_get_tos_config",
        lambda: {
            "ak": "test-ak",
            "sk": "test-sk",
            "bucket": "honcut-fixtures",
            "endpoint": "tos-cn-beijing.volces.com",
            "region": "cn-beijing",
        },
    )


class _FakeTOSClient:
    def __init__(self, *, head_responses, put_response=None, put_error=None):
        self.head_responses = list(head_responses)
        self.put_response = put_response
        self.put_error = put_error
        self.head_calls = 0
        self.put_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def head(self, *_args, **_kwargs):
        self.head_calls += 1
        if not self.head_responses:
            raise AssertionError("unexpected TOS HEAD")
        response = self.head_responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def put(self, *_args, **_kwargs):
        self.put_calls += 1
        if self.put_error is not None:
            raise self.put_error
        return self.put_response


def _missing_object():
    return SimpleNamespace(status_code=404, headers={}, content=b"")


def _matching_object(payload: bytes):
    return SimpleNamespace(
        status_code=200,
        headers={
            "Content-Length": str(len(payload)),
            tos_uploader.TOS_CONTENT_SHA256_METADATA: hashlib.sha256(payload).hexdigest(),
        },
        content=b"",
    )


def _policy() -> TOSUploadExecutionPolicy:
    return TOSUploadExecutionPolicy(
        connect_timeout_seconds=3,
        read_timeout_seconds=5,
        minimum_write_timeout_seconds=10,
        maximum_write_timeout_seconds=30,
        write_overhead_seconds=2,
        minimum_upload_bytes_per_second=1024,
        pool_timeout_seconds=4,
        reconciliation_timeout_seconds=6,
    )


def test_tos_upload_policy_scales_and_bounds_write_timeout():
    policy = _policy()

    small = policy.timeouts_for_payload(100)
    medium = policy.timeouts_for_payload(12 * 1024)
    large = policy.timeouts_for_payload(100 * 1024)

    assert small.write_seconds == 10
    assert medium.write_seconds == 14
    assert large.write_seconds == 30
    assert small.connect_seconds == 3
    assert small.reconciliation_seconds == 6


def test_tos_runtime_upload_policy_is_process_wide_and_non_nested(tmp_path):
    fallback = MediaUploadTimeouts(1, 1, 1, 1, 1)

    with tos_upload_execution_scope(tmp_path, policy=_policy()):
        with ThreadPoolExecutor(max_workers=1) as executor:
            resolved = executor.submit(
                effective_media_upload_timeouts,
                12 * 1024,
                fallback=fallback,
            ).result()
        assert resolved.write_seconds == 14
        with pytest.raises(RuntimeError, match="already active"):
            with tos_upload_execution_scope(tmp_path, policy=_policy()):
                pass

    assert effective_media_upload_timeouts(12 * 1024, fallback=fallback) == fallback


def test_tos_write_timeout_reconciles_by_head_without_second_put(
    monkeypatch,
    tmp_path,
):
    _configure_tos(monkeypatch)
    payload = b"reconciled-payload"
    payload_hash = hashlib.sha256(payload).hexdigest()
    object_key = f"volcengine/multimodal/{payload_hash}.bin"
    request = httpx.Request("PUT", "https://tos.invalid/object")
    transport = _FakeTOSClient(
        head_responses=[_missing_object(), _matching_object(payload)],
        put_error=httpx.WriteTimeout("write stalled", request=request),
    )
    monkeypatch.setattr(
        tos_uploader,
        "_new_tos_http_client",
        lambda _timeouts: transport,
    )

    with tos_upload_execution_scope(tmp_path, policy=_policy()):
        signed_url = tos_uploader.upload_file_required(
            payload,
            object_key,
            "application/octet-stream",
        )

    assert object_key in signed_url
    assert transport.put_calls == 1
    assert transport.head_calls == 2
    ledger = json.loads((tmp_path / TOS_UPLOAD_LEDGER_NAME).read_text())
    assert ledger["submission_attempt_count"] == 1
    assert ledger["uploads"][0]["status"] == "reconciled_completed"
    assert [event["event"] for event in ledger["uploads"][0]["transitions"]] == [
        "UploadPrepared",
        "SubmissionAttempted",
        "UploadReconciled",
    ]


def test_tos_success_without_exact_post_head_remains_uncertain(
    monkeypatch,
    tmp_path,
):
    _configure_tos(monkeypatch)
    payload = b"accepted-but-unverified"
    payload_hash = hashlib.sha256(payload).hexdigest()
    object_key = f"volcengine/multimodal/{payload_hash}.bin"
    transport = _FakeTOSClient(
        head_responses=[_missing_object(), _missing_object()],
        put_response=SimpleNamespace(status_code=200, headers={}, content=b""),
    )
    monkeypatch.setattr(
        tos_uploader,
        "_new_tos_http_client",
        lambda _timeouts: transport,
    )

    with tos_upload_execution_scope(tmp_path, policy=_policy()):
        with pytest.raises(tos_uploader.TOSUploadOutcomeUnknown):
            tos_uploader.upload_file_required(
                payload,
                object_key,
                "application/octet-stream",
            )

    assert transport.put_calls == 1
    assert transport.head_calls == 2
    ledger = json.loads((tmp_path / TOS_UPLOAD_LEDGER_NAME).read_text())
    assert ledger["submission_attempt_count"] == 1
    assert ledger["uploads"][0]["status"] == "submission_uncertain"


def test_tos_preexisting_object_without_hash_metadata_is_not_reused(
    monkeypatch,
    tmp_path,
):
    _configure_tos(monkeypatch)
    payload = b"metadata-required"
    payload_hash = hashlib.sha256(payload).hexdigest()
    object_key = f"volcengine/multimodal/{payload_hash}.bin"
    length_only = SimpleNamespace(
        status_code=200,
        headers={"Content-Length": str(len(payload))},
        content=b"",
    )
    transport = _FakeTOSClient(
        head_responses=[length_only, _matching_object(payload)],
        put_response=SimpleNamespace(status_code=200, headers={}, content=b""),
    )
    monkeypatch.setattr(
        tos_uploader,
        "_new_tos_http_client",
        lambda _timeouts: transport,
    )

    with tos_upload_execution_scope(tmp_path, policy=_policy()):
        tos_uploader.upload_file_required(
            payload,
            object_key,
            "application/octet-stream",
        )

    assert transport.put_calls == 1


def test_tos_unknown_timeout_never_reputs_on_resume(monkeypatch, tmp_path):
    _configure_tos(monkeypatch)
    payload = b"unresolved-payload"
    payload_hash = hashlib.sha256(payload).hexdigest()
    object_key = f"volcengine/multimodal/{payload_hash}.bin"
    request = httpx.Request("PUT", "https://tos.invalid/object")
    first_transport = _FakeTOSClient(
        head_responses=[_missing_object(), _missing_object()],
        put_error=httpx.WriteTimeout("write stalled", request=request),
    )
    monkeypatch.setattr(
        tos_uploader,
        "_new_tos_http_client",
        lambda _timeouts: first_transport,
    )

    with tos_upload_execution_scope(tmp_path, policy=_policy()):
        with pytest.raises(tos_uploader.TOSUploadOutcomeUnknown):
            tos_uploader.upload_file_required(
                payload,
                object_key,
                "application/octet-stream",
            )

    second_transport = _FakeTOSClient(head_responses=[_missing_object()])
    monkeypatch.setattr(
        tos_uploader,
        "_new_tos_http_client",
        lambda _timeouts: second_transport,
    )
    with tos_upload_execution_scope(tmp_path, policy=_policy()):
        with pytest.raises(tos_uploader.TOSUploadOutcomeUnknown):
            tos_uploader.upload_file_required(
                payload,
                object_key,
                "application/octet-stream",
            )

    assert first_transport.put_calls == 1
    assert second_transport.put_calls == 0
    ledger = json.loads((tmp_path / TOS_UPLOAD_LEDGER_NAME).read_text())
    assert ledger["submission_attempt_count"] == 1
    assert ledger["uploads"][0]["status"] == "submission_uncertain"


def test_tos_known_rejection_is_typed_and_not_retried(monkeypatch, tmp_path):
    _configure_tos(monkeypatch)
    payload = b"rejected-payload"
    payload_hash = hashlib.sha256(payload).hexdigest()
    object_key = f"volcengine/multimodal/{payload_hash}.bin"
    transport = _FakeTOSClient(
        head_responses=[_missing_object()],
        put_response=SimpleNamespace(
            status_code=403,
            headers={},
            content=b"credentials must not be copied into receipts",
        ),
    )
    monkeypatch.setattr(
        tos_uploader,
        "_new_tos_http_client",
        lambda _timeouts: transport,
    )

    with tos_upload_execution_scope(tmp_path, policy=_policy()):
        with pytest.raises(tos_uploader.TOSUploadRejected, match="HTTP 403"):
            tos_uploader.upload_file_required(
                payload,
                object_key,
                "application/octet-stream",
            )

    assert transport.put_calls == 1
    ledger_path = tmp_path / TOS_UPLOAD_LEDGER_NAME
    ledger_text = ledger_path.read_text()
    ledger = json.loads(ledger_text)
    assert ledger["uploads"][0]["status"] == "provider_rejected"
    assert "credentials must not" not in ledger_text


def test_tos_completed_receipt_fails_closed_when_remote_object_disappears(
    monkeypatch,
    tmp_path,
):
    _configure_tos(monkeypatch)
    payload = b"completed-payload"
    payload_hash = hashlib.sha256(payload).hexdigest()
    object_key = f"volcengine/multimodal/{payload_hash}.bin"
    first_transport = _FakeTOSClient(
        head_responses=[_missing_object(), _matching_object(payload)],
        put_response=SimpleNamespace(status_code=200, headers={}, content=b""),
    )
    monkeypatch.setattr(
        tos_uploader,
        "_new_tos_http_client",
        lambda _timeouts: first_transport,
    )
    with tos_upload_execution_scope(tmp_path, policy=_policy()):
        tos_uploader.upload_file_required(
            payload,
            object_key,
            "application/octet-stream",
        )

    missing_transport = _FakeTOSClient(head_responses=[_missing_object()])
    monkeypatch.setattr(
        tos_uploader,
        "_new_tos_http_client",
        lambda _timeouts: missing_transport,
    )
    with tos_upload_execution_scope(tmp_path, policy=_policy()):
        with pytest.raises(tos_uploader.TOSReceiptMismatch):
            tos_uploader.upload_file_required(
                payload,
                object_key,
                "application/octet-stream",
            )

    assert first_transport.put_calls == 1
    assert missing_transport.put_calls == 0


def test_tos_http_client_ignores_ambient_proxy(monkeypatch):
    observed = {}

    class _Client:
        def __init__(self, **kwargs):
            observed.update(kwargs)

    monkeypatch.setattr(tos_uploader.httpx, "Client", _Client)
    timeouts = _policy().timeouts_for_payload(1024)

    tos_uploader._new_tos_http_client(timeouts)

    assert observed["trust_env"] is False
    timeout = observed["timeout"]
    assert timeout.write == timeouts.write_seconds
    assert timeout.connect == timeouts.connect_seconds


def test_tos_hard_limit_blocks_second_put_before_network(monkeypatch, tmp_path):
    _configure_tos(monkeypatch)
    first = b"first"
    second = b"second"
    transport = _FakeTOSClient(
        head_responses=[
            _missing_object(),
            _matching_object(first),
            _missing_object(),
        ],
        put_response=SimpleNamespace(status_code=200, headers={}, content=b""),
    )
    monkeypatch.setattr(
        tos_uploader,
        "_new_tos_http_client",
        lambda _timeouts: transport,
    )

    with tos_upload_execution_scope(
        tmp_path,
        policy=_policy(),
        max_submissions=1,
    ):
        tos_uploader.upload_file_required(
            first,
            f"volcengine/media/{hashlib.sha256(first).hexdigest()}.bin",
            "application/octet-stream",
        )
        with pytest.raises(RuntimeError, match="hard limit"):
            tos_uploader.upload_file_required(
                second,
                f"volcengine/media/{hashlib.sha256(second).hexdigest()}.bin",
                "application/octet-stream",
            )

    assert transport.put_calls == 1


def test_tos_future_ledger_schema_fails_before_network(monkeypatch, tmp_path):
    _configure_tos(monkeypatch)
    (tmp_path / TOS_UPLOAD_LEDGER_NAME).write_text(
        json.dumps(
            {
                "schema": "honcut.tos-upload-ledger.v999",
                "hard_limit": None,
                "submission_attempt_count": 0,
                "uploads": [],
            }
        )
    )
    transport = _FakeTOSClient(head_responses=[])
    monkeypatch.setattr(
        tos_uploader,
        "_new_tos_http_client",
        lambda _timeouts: transport,
    )
    payload = b"future-ledger"

    with tos_upload_execution_scope(tmp_path, policy=_policy()):
        with pytest.raises(RuntimeError, match="schema is unsupported"):
            tos_uploader.upload_file_required(
                payload,
                f"volcengine/media/{hashlib.sha256(payload).hexdigest()}.bin",
                "application/octet-stream",
            )

    assert transport.head_calls == 0
    assert transport.put_calls == 0


def test_content_address_uses_final_transmitted_bytes(monkeypatch, tmp_path):
    _configure_tos(monkeypatch)
    source = b"source-image-bytes"
    transmitted = b"final-transmitted-bytes"
    final_hash = hashlib.sha256(transmitted).hexdigest()
    source_hash = hashlib.sha256(source).hexdigest()
    transport = _FakeTOSClient(
        head_responses=[_missing_object(), _matching_object(transmitted)],
        put_response=SimpleNamespace(status_code=200, headers={}, content=b""),
    )
    monkeypatch.setattr(
        tos_uploader,
        "compress_image_bytes",
        lambda _payload: transmitted,
    )
    monkeypatch.setattr(
        tos_uploader,
        "_new_tos_http_client",
        lambda _timeouts: transport,
    )

    with tos_upload_execution_scope(tmp_path, policy=_policy()):
        signed_url = tos_uploader.upload_file_required(
            source,
            f"volcengine/media/{source_hash}.png",
            "image/png",
        )

    assert final_hash in signed_url
    assert source_hash not in signed_url
    ledger = json.loads((tmp_path / TOS_UPLOAD_LEDGER_NAME).read_text())
    assert ledger["uploads"][0]["payload"]["payload_sha256"] == final_hash


def test_paid_media_owners_do_not_use_optional_tos_upload_fallbacks():
    repo_root = Path(__file__).resolve().parents[2]
    owners = (
        repo_root / "pipeline/src/runtime/continuity_provider.py",
        repo_root / "pipeline/src/tools/asset_packager.py",
        repo_root / "pipeline/src/clients/ark_multimodal_client.py",
        repo_root / "pipeline/src/clients/seedance_client.py",
        repo_root / "pipeline/src/phases/phase6/direct_generation.py",
    )
    forbidden = (
        "tos_uploader.upload_file(",
        "tos_uploader.upload_image(",
        "tos_uploader.upload_media_file(",
        "from clients.tos_uploader import upload_image\n",
        "from clients.tos_uploader import upload_media_file\n",
    )

    for owner in owners:
        source = owner.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), owner
