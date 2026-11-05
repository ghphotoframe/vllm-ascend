# FlashKDA for Ling prefill on A5

The optional PyPTO FlashKDA V1 operator replaces AscendC chunk KDA for
Bailing/Ling V3 prefill. It is disabled by default. Kimi and GLM continue to
use the shared AscendC implementation. Recurrent KDA, including decode and
speculative verification, continues to use AscendC.

## Enable and disable

Set the environment variable before starting the server and all its workers:

```bash
# Existing AscendC chunk KDA (default on every supported device).
export VLLM_ASCEND_ENABLE_FLASH_KDA=0

# Optional FlashKDA prefill on A5.
export VLLM_ASCEND_ENABLE_FLASH_KDA=1
```

Restart workers after changing the value. Only `0` and `1` are accepted.
Explicit enablement on A2, A3 or 310P raises an error before importing the
FlashKDA adapter. Hardware support uses the device profile; the worker's
existing runtime check verifies that the installed package matches its NPU.

`ops/kda_backend.py` selects an implementation at import time and exposes the
fixed name `run_chunk_kda`. The model's prefill call site and state
gather/scatter therefore remain shared between the two implementations.
Unsupported inputs and missing dependencies raise errors; they do not
silently fall back during an A/B comparison.

## External dependency

Obtain the vendor's FlashKDA V1 source and compatible PyPTO wheel separately.
Put the directory containing the vendor's `flash_kda/` directory on
`PYTHONPATH`, so this import succeeds in the worker environment:

```python
from flash_kda.flash_kda import flash_kda_wrapper
```

The delivery guide specifies A5, x86_64, Python 3.12, HDK 25.5.x and PyPTO
0.2.1, with a CANN 9.2.0 bundle. Compatibility with other CANN/runtime
versions must be verified in an isolated environment. Do not replace a
shared CANN installation to enable this option. The package is optional;
default deployments do not import FlashKDA or PyPTO.

The adapter has been checked against the delivered V1 Python interface.
Runtime compatibility and numerical results must still be verified before
enabling it in a serving deployment. The delivery report uses torch 2.13.0
and torch_npu 2.13.0; validate the supplied wheel against the serving image
in a separate process before loading it into a worker.

## Tensor and numerical contract

- Packed BSND input with batch size 1, positive sequence lengths and
  `Dk = Dv = 128`; `Nv % Nk == 0`, `Nk <= 64`, `Nv <= 96`, and at most
  64 sequences per call. Larger packed batches are rejected explicitly.
- BF16 Q/K/V, raw gate and raw beta logits on the same NPU.
- Bounded gate enabled, with `lower_bound` in `[-5, 0]`.
- State layout `[num_sequences, Nv, Dv, Dk]`. BF16 cache rows are promoted
  to FP32 at the adapter boundary. Existing scatter handles cache dtype.
- `A_log` is packed as FP32 `[Nv]`; `dt_bias` as FP32 `[Nv, 128]`.
- Host sequence offsets come from the GDN builder after empty-sequence
  filtering and are uploaded as contiguous NPU int64 offsets. V1 accepts
  int32 too, but converts it to int64 on device; supplying int64 avoids that
  extra cast. The primary varlen entry does not read device offsets back to
  the CPU.

FlashKDA performs L2 normalization, beta sigmoid and gate conversion inside
the kernel. The Ling forward path preserves the original beta logits when
this backend is selected. The recurrent path applies the original FP32
sigmoid before calling AscendC. No inverse-sigmoid reconstruction is used.

The adapter always supplies `cu_seqlens`, including for a single sequence
whose length is not divisible by 64. Tail handling belongs to the vendor
kernel; no input padding is added by this integration.

## Validation before use

Compare output and final state against AscendC, then test state continuation
across chunked prefill, decode, prefix-cache reuse and speculative rollback.
Include non-contiguous projection views and BF16/FP32 state caches.
Use the current production helper as the reference: it normalizes Q/K and
computes FP32 beta sigmoid outside the chunk kernel. The delivery report's
comparison with in-kernel preprocessing does not establish equivalence to
this exact serving path.

For performance, measure both the vendor kernel and the complete adapter,
including sequence-offset upload, contiguous copies and state conversion.
Separate first-use compilation from warmed execution. Finally compare
end-to-end TTFT, prefill throughput, TPOT and output correctness with an
otherwise identical serving configuration.

The author's optimization focus on `Nk = Nv = 4` and 28 cores is not a
guarantee for other shapes or a declared support restriction. Ling's local
head count depends on tensor parallelism; collect the actual per-rank shapes
and sequence-length distributions before deciding whether tuning is needed.
V1 queries core counts through PyPTO's platform API and selects its stage2
tiling from those counts; it does not hard-code a 28-core launch.
