# FlashKDA for Ling prefill on A5

The optional PyPTO FlashKDA V1 implementation replaces AscendC chunk KDA for
Bailing/Ling V3 prefill. Its kernel definitions are included in the
vLLM Ascend source and installed package. It is experimental and disabled
by default. Kimi and GLM continue to use the shared AscendC implementation.
Recurrent KDA, including decode and speculative verification, continues to
use AscendC.

## Enable and disable

Set the environment variable before starting the server and all its workers:

```bash
# Existing AscendC chunk KDA (default on every supported device).
export VLLM_ASCEND_ENABLE_FLASH_KDA=0

# Optional FlashKDA prefill on A5.
export VLLM_ASCEND_ENABLE_FLASH_KDA=1
```

Restart all workers after changing the value, the installed FlashKDA source,
or the PyPTO runtime and its patches. Only `0` and `1` are accepted.
Explicit enablement on A2, A3 or 310P raises an error before importing the
FlashKDA adapter. Hardware support uses the device profile; the worker's
existing runtime check verifies that the installed package matches its NPU.

`ops/kda_backend.py` selects an implementation at import time and exposes the
fixed name `run_chunk_kda`. The model's prefill call site and state
gather/scatter therefore remain shared between the two implementations.
Unsupported inputs and missing dependencies raise errors; they do not
silently fall back during an A/B comparison.

## Implementation and runtime dependencies

FlashKDA is provided by `vllm_ascend/ops/pypto/flash_kda.py`. The serving
adapter lazily imports the wrapper from this installed module:

```python
from vllm_ascend.ops.pypto.flash_kda import flash_kda_wrapper
```

No separate FlashKDA checkout, external `flash_kda` Python package, or
FlashKDA-specific `PYTHONPATH` entry is required. The default AscendC path
does not import this kernel module or PyPTO; enabling the backend keeps
the operator/runtime import deferred until a worker executes FlashKDA prefill.

PyPTO remains a separate optional runtime dependency. The reference
environment is locked to the compatible PyPTO **0.2.1** wheel, which supplies
`pypto_pro` and its compiler/runtime components. The in-tree implementation
does not vendor or install that wheel. Match its architecture and Python ABI
to the serving image and verify the complete Torch, torch_npu, CANN and
driver combination in an isolated process. A matching PyPTO version string
alone does not establish runtime compatibility.

The PyPTO debug-command caching fix is also a separate runtime change in
`pypto_pro/runtime/exception_dump.py`. It is not applied by importing or
installing vLLM Ascend. Reproducing the optimized reference configuration
requires that independently validated runtime fix as well as the batch-owned
sequence-offset reuse described below. Use a private runtime copy and record
its actual import path and patch fingerprint. A private PyPTO import path is
independent of the bundled FlashKDA module path.

The runtime fix caches invariant debug-command construction while retaining
per-launch dynamic tensor/scalar ABI records and native setters. It must not
disable exception diagnostics or freeze changing addresses, shapes or launch
parameters. Runtime/source replacement requires cache invalidation at a safe
boundary or a worker restart; do not mutate a shared serving environment to
apply the fix.

## Tensor and numerical contract

- Packed BSND input with batch size 1, positive sequence lengths and
  `Dk = Dv = 128`; `Nv % Nk == 0`, `Nk <= 64`, `Nv <= 96`, and at most
  64 sequences per call. Larger packed batches are rejected explicitly.
- BF16 Q/K/V, raw gate and raw beta logits on the same NPU.
- Bounded gate enabled, with `lower_bound` in `[-5, 0]`.
- State layout `[num_sequences, Nv, Dv, Dk]`. BF16 cache rows are promoted
  to FP32 at the adapter boundary. Existing scatter handles cache dtype.
- `A_log` is packed as FP32 `[Nv]`; `dt_bias` as FP32 `[Nv, 128]`.
- Host sequence offsets come from the GDN builder after prefill/decode
  splitting and empty-sequence filtering. The FlashKDA metadata builder
  creates a pinned CPU int64 descriptor and its contiguous NPU int64 copy
  once per batch. Batch metadata retains both tensors, and every KDA layer
  reuses the same prepared device descriptor through `cu_seqlens_device`.
  The adapter checks its dtype, device, shape and contiguity without reading
  device values. The default AscendC builder does not create this additional
  FlashKDA descriptor. Supplying int64 avoids V1's int32-to-int64 device cast.

FlashKDA performs L2 normalization, beta sigmoid and gate conversion inside
the kernel. The Ling forward path preserves the original beta logits when
this backend is selected. The recurrent path applies the original FP32
sigmoid before calling AscendC. No inverse-sigmoid reconstruction is used.

The adapter always supplies `cu_seqlens`, including for a single sequence
whose length is not divisible by 64. Tail handling belongs to the in-tree
kernel; the adapter does not pad logical inputs. Internal kernel workspaces
can have padded shapes.

Direct operator tools can omit `cu_seqlens_device` and let the adapter
construct a descriptor for that call. Serving supplies the batch-owned
tensor, avoiding per-layer descriptor allocation and host-to-device transfer.
Prepared device values must match the validated host sequence offsets;
this is the caller's contract, and the adapter does not read device values
to check it.

## Validation before use

Numerical accuracy acceptance remains **`NOT_ACCEPTED`**. Bundling the
implementation and reducing host overhead do not resolve the existing
output/final-state validation gaps. Neither successful requests nor an
unchanged result on a few captured inputs establishes serving accuracy.

Compare output and final state against AscendC, then test state continuation
across chunked prefill, decode, prefix-cache reuse and speculative rollback.
Include non-contiguous projection views and BF16/FP32 state caches.
Use the current production helper as the reference: it normalizes Q/K and
computes FP32 beta sigmoid outside the chunk kernel. The delivery report's
comparison with in-kernel preprocessing does not establish equivalence to
this exact serving path.

For performance, measure both the in-tree kernel and the complete adapter,
including sequence-offset upload, contiguous copies and state conversion.
Separate first-use PyPTO compilation from warmed execution. Run a complete
workload warmup before a separate formal measurement, using distinct cache
salts while retaining the same within-run prefix-cache policy. Compare
end-to-end TTFT, prefill throughput and output correctness with otherwise
identical serving configurations and actual per-rank inputs. One output
token does not provide a meaningful decode TPOT measurement.

Earlier isolated measurements used one formal 200-request sample per backend
after warmup. They do not establish a stable approximately 3% gain across
repeated runs, other shapes or other environments. Repeat measurements and
retain cache-hit, compilation and runtime-fingerprint evidence before making
a performance claim. The combined sequence-offset and PyPTO runtime changes
also do not isolate the contribution of either change individually.

The author's optimization focus on `Nk = Nv = 4` and 28 cores is not a
guarantee for other shapes or a declared support restriction. Ling's local
head count depends on tensor parallelism; collect the actual per-rank shapes
and sequence-length distributions before deciding whether tuning is needed.
V1 queries core counts through PyPTO's platform API and selects its stage2
tiling from those counts; it does not hard-code a 28-core launch.
