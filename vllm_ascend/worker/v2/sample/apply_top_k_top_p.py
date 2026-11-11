from functools import wraps

import numpy as np
import torch

from vllm_ascend.sample.sampler import _apply_top_k_top_p_pytorch

# Limit selection work; larger supports and top-p retain the validated sort path.
_MAX_FAST_TOP_K = 64


def apply_top_k_top_p_npu(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    *,
    max_top_k: int | None = None,
) -> torch.Tensor:
    if k is None and p is None:
        return logits
    if (
        k is not None
        and p is None
        and max_top_k is not None
        and 0 < max_top_k <= min(_MAX_FAST_TOP_K, logits.shape[-1])
    ):
        # Keep the existing probability-domain cutoff, including ties and
        # softmax underflow. A logits-domain cutoff would change those masks.
        # The bound comes from existing CPU request metadata, never k.max().
        probs = logits.softmax(dim=-1)
        no_top_k = k == logits.shape[-1]
        indices = (k.masked_fill(no_top_k, 1).to(torch.long) - 1).unsqueeze(-1)
        cutoff = probs.topk(max_top_k, dim=-1).values.gather(-1, indices)
        cutoff.masked_fill_(no_top_k.unsqueeze(-1), -float("inf"))
        return logits.masked_fill_(probs < cutoff, -float("inf"))
    return _apply_top_k_top_p_pytorch(logits, k, p)


def apply_top_k_top_p_with_cpu_metadata(
    self,
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
) -> torch.Tensor:
    """V2 SamplingStates entry; preserve the common fallback and RNG path."""
    top_k, top_p = self.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
    if top_k is None and top_p is None:
        return logits
    max_top_k = None
    if top_k is not None and top_p is None:
        active_k = self.top_k.np[idx_mapping_np]
        bounded_k = active_k[active_k != self.vocab_size]
        if bounded_k.size:
            max_top_k = int(bounded_k.max())
    return apply_top_k_top_p_npu(logits, top_k, top_p, max_top_k=max_top_k)


def wrap_sample_random_with_cpu_metadata(original_sample_random):
    """Carry V2's CPU bound into its direct (non-states) sampling entry."""

    @wraps(original_sample_random)
    def sample_random(self, processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p, use_flashinfer):
        if not use_flashinfer and top_k is not None and top_p is None:
            active_k = self.sampling_states.top_k.np[idx_mapping_np]
            bounded_k = active_k[active_k != self.sampling_states.vocab_size]
            max_top_k = int(bounded_k.max()) if bounded_k.size else 0
            if 0 < max_top_k <= min(_MAX_FAST_TOP_K, processed_logits.shape[-1]):
                processed_logits = apply_top_k_top_p_npu(processed_logits, top_k, None, max_top_k=max_top_k)
                # The original method still owns categorical sampling, seed /
                # position handling and processed-logits output. Its filter
                # becomes a no-op because the identical mask is already applied.
                return original_sample_random(
                    self, processed_logits, expanded_idx_mapping, idx_mapping_np, pos, None, None, False
                )
        return original_sample_random(
            self, processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p, use_flashinfer
        )

    return sample_random
