# SPDX-License-Identifier: Apache-2.0
"""Keep A5 provider callbacks mapped across HCCL communication groups."""

import ctypes
import os


def pin_a5_rdma_provider():
    """Keep provider callbacks valid across HCCL groups when A5 is present."""
    provider_path = "/usr/lib64/libhrn5-rdmav34.so"
    if not os.path.isfile(provider_path):
        return None
    return ctypes.CDLL(provider_path, mode=os.RTLD_NOW | os.RTLD_NODELETE)
