# SPDX-License-Identifier: Apache-2.0
"""A5 provider lifetime pinning must not break hosts without the provider."""

from unittest.mock import Mock

from vllm_ascend.distributed import rdma_provider as provider


def test_missing_provider_does_not_load_a_library(monkeypatch):
    load = Mock()
    monkeypatch.setattr(provider.os.path, "isfile", lambda path: False)
    monkeypatch.setattr(provider.ctypes, "CDLL", load)
    assert provider.pin_a5_rdma_provider() is None
    load.assert_not_called()


def test_existing_provider_keeps_nodelete_handle(monkeypatch):
    handle = object()
    load = Mock(return_value=handle)
    monkeypatch.setattr(provider.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(provider.ctypes, "CDLL", load)
    assert provider.pin_a5_rdma_provider() is handle
    load.assert_called_once_with("/usr/lib64/libhrn5-rdmav34.so", mode=provider.os.RTLD_NOW | provider.os.RTLD_NODELETE)
