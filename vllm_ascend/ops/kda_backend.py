# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Import-time prefill selection for Bailing/Ling V3.

FlashKDA consumes raw beta logits, whereas the AscendC path consumes FP32
sigmoid(beta). The Ling caller preserves that contract through this immutable
selection. Other KDA models continue to import the shared AscendC helpers.
"""

from vllm_ascend import envs
from vllm_ascend.device import HardwareCapability, get_current_hardware_profile

FLASH_KDA_ENABLED = envs.VLLM_ASCEND_ENABLE_FLASH_KDA

if FLASH_KDA_ENABLED:
    if not get_current_hardware_profile().supports(HardwareCapability.FLASH_KDA_PREFILL):
        raise RuntimeError("VLLM_ASCEND_ENABLE_FLASH_KDA=1 requires A5; disable it on A2/A3/310P.")
    from vllm_ascend.ops.flash_kda import run_flash_chunk_kda as run_chunk_kda
else:
    from vllm_ascend.ops.kda import run_chunk_kda as run_chunk_kda

__all__ = ["FLASH_KDA_ENABLED", "run_chunk_kda"]
