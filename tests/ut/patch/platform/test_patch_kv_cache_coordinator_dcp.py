# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise replicated Mamba states with real upstream cache managers/pool.

No tensors or device operations are needed: these tests verify cache geometry,
hash ownership and the copy plans consumed by the model runner.

The 128-token hash geometry was the initial hypothesis, not a measured runtime
value. Also cover 256-token hashes/Mamba blocks, which reproduce the observed
8083-token prompt's 7936-token reuse. CacheConfig metrics alone do not establish
the resolved scheduler group specs.
"""

from math import lcm
from types import SimpleNamespace

import pytest
import torch
import vllm.v1.core.kv_cache_manager as manager_module
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlockCopy,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

from vllm_ascend.patch.platform import patch_kv_cache_coordinator as mod
from vllm_ascend.patch.platform.patch_mamba_block_aligned_split import (
    _original_mamba_block_aligned_split,
)

HASH_BLOCK_SIZE = 128


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def _request(request_id, prompt_length=8083, cache_salt="shared", hash_block_size=HASH_BLOCK_SIZE):
    sampling_params = SamplingParams(max_tokens=64)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_length)),
        sampling_params=sampling_params,
        pooling_params=None,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(hash_block_size, sha256),
    )


def _manager(monkeypatch, *, dcp=2, mamba_block=128, allow_partial=True, hash_block_size=HASH_BLOCK_SIZE):
    config = KVCacheConfig(
        num_blocks=256,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(block_size=256, num_kv_heads=1, head_size=1, dtype=torch.float32),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=mamba_block,
                    shapes=((1,),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )

    def coordinator_factory(**kwargs):
        kwargs["allow_partial_hash_hits"] = allow_partial
        return mod.get_kv_cache_coordinator(**kwargs)

    monkeypatch.setattr(manager_module, "get_kv_cache_coordinator", coordinator_factory)
    manager = manager_module.KVCacheManager(
        config,
        max_model_len=16384,
        scheduler_block_size=lcm(256 * dcp, mamba_block),
        hash_block_size=hash_block_size,
        dcp_world_size=dcp,
        enable_caching=True,
    )
    # Exercise the upstream retention mask, not only the None fast path.
    manager.coordinator.retention_interval = 0
    return manager


@pytest.mark.parametrize("dcp", [1, 2])
@pytest.mark.parametrize("mamba_block", [128, 256])
@pytest.mark.parametrize("allow_partial", [False, True])
def test_partial_hit_admission_and_per_spec_geometry(monkeypatch, dcp, mamba_block, allow_partial):
    manager = _manager(monkeypatch, dcp=dcp, mamba_block=mamba_block, allow_partial=allow_partial)
    coordinator = manager.coordinator
    attention, mamba = coordinator.single_type_managers
    assert (attention.dcp_world_size, attention.block_size) == (dcp, 256 * dcp)
    assert (mamba.dcp_world_size, mamba.block_size) == (1, mamba_block)
    enabled = allow_partial and (dcp > 1 or mamba_block > HASH_BLOCK_SIZE)
    assert coordinator.enable_partial_hash_hits is enabled
    alignment = HASH_BLOCK_SIZE if enabled else lcm(256 * dcp, mamba_block)
    assert coordinator._cache_hit_alignment_tokens == alignment
    assert attention.cache_hit_alignment_tokens == mamba.cache_hit_alignment_tokens == alignment


def test_unsupported_partial_lookup_manager_disables_fine_hits(monkeypatch):
    original_factory = mod.get_manager_for_kv_cache_spec

    def manager_factory(**kwargs):
        manager = original_factory(**kwargs)
        if isinstance(kwargs["kv_cache_spec"], FullAttentionSpec):
            manager.supports_fine_grained_hash_lookup = False
        return manager

    monkeypatch.setattr(mod, "get_manager_for_kv_cache_spec", manager_factory)
    manager = _manager(monkeypatch)
    assert not manager.coordinator.enable_partial_hash_hits
    assert manager.coordinator._cache_hit_alignment_tokens == 512


def test_other_ascend_cache_types_keep_existing_dcp_geometry():
    spec = SlidingWindowSpec(block_size=256, num_kv_heads=1, head_size=1, dtype=torch.float32, sliding_window=512)
    assert mod._manager_dcp_world_size(spec, 2) == 2


@pytest.mark.parametrize(
    ("hash_block_size", "expected_boundaries"),
    [(128, [7936, 8064, 8083]), (256, [7936, 8083])],
)
def test_dcp_prompt_tail_split_reaches_reusable_hash_boundary(monkeypatch, hash_block_size, expected_boundaries):
    manager = _manager(monkeypatch, hash_block_size=hash_block_size, mamba_block=hash_block_size)
    scheduler = SimpleNamespace(
        block_size=512,
        cache_config=SimpleNamespace(block_size=256),
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=False,
        use_eagle_block_drop=False,
        hash_block_size=hash_block_size,
        dcp_world_size=2,
        scheduler_block_size=512,
        mamba_partial_cache_hit=manager.coordinator.enable_partial_hash_hits,
        mamba_fine_grained_prefix_cache=False,
        mamba_has_prefill_checkpoint_blocks=False,
    )
    request = _request("split", hash_block_size=hash_block_size)
    boundaries = []
    while request.num_computed_tokens < request.num_tokens:
        scheduled = _original_mamba_block_aligned_split(
            scheduler, request, num_new_tokens=request.num_tokens - request.num_computed_tokens
        )
        assert scheduled > 0
        request.num_computed_tokens += scheduled
        boundaries.append(request.num_computed_tokens)
    assert boundaries == expected_boundaries


@pytest.mark.parametrize(("hash_block_size", "snapshot"), [(128, 8064), (256, 7936)])
def test_dcp_reuses_snapshot_and_copies_partial_attention_tail(monkeypatch, hash_block_size, snapshot):
    manager = _manager(monkeypatch, hash_block_size=hash_block_size, mamba_block=hash_block_size)
    coordinator = manager.coordinator
    attention, mamba = coordinator.single_type_managers
    assert coordinator.enable_partial_hash_hits
    assert coordinator._cache_hit_alignment_tokens == hash_block_size
    assert (attention.block_size, attention.dcp_world_size) == (512, 2)
    assert (mamba.block_size, mamba.dcp_world_size) == (hash_block_size, 1)
    owner = _request("owner", hash_block_size=hash_block_size)
    empty, hit, _ = manager.get_computed_blocks(owner)
    assert hit == 0
    assert manager.allocate_slots(owner, snapshot, hit, empty) is not None
    manager.free(owner)
    manager.new_step_starts()

    consumer = _request("consumer", hash_block_size=hash_block_size)
    blocks, hit, _ = manager.get_computed_blocks(consumer)
    assert hit == snapshot
    assert [len(group) for group in blocks.blocks] == [16, snapshot // hash_block_size]
    _, per_group_hits = manager.coordinator.find_longest_cache_hit_per_group(
        consumer.block_hashes, consumer.num_tokens - 1
    )
    assert per_group_hits == (snapshot, snapshot)
    attention_tail, mamba_state = (group[-1] for group in blocks.blocks)
    original_hashes = (attention_tail.block_hash, mamba_state.block_hash)
    assert (
        manager.get_computed_blocks(_request("different-salt", cache_salt="other", hash_block_size=hash_block_size))[1]
        == 0
    )

    new_blocks = manager.allocate_slots(consumer, consumer.num_tokens - snapshot, hit, blocks)
    assert new_blocks is not None
    attention_copy_id = new_blocks.get_block_ids()[0][0]
    copies, retained = manager.take_kv_cache_block_copies()
    assert KVCacheBlockCopy(attention_tail.block_id, attention_copy_id) in copies
    # Each snapshot ends a complete replicated Mamba block: resume into the next
    # slot, while the shared attention tail needs CoW inside its 512-token page.
    assert all(copy.src_block_id != mamba_state.block_id for copy in copies)
    assert (attention_tail.block_hash, mamba_state.block_hash) == original_hashes
    manager.block_pool.free_blocks(retained)
    manager.free(consumer)
    manager.new_step_starts()
    assert manager.get_computed_blocks(_request("repeat", hash_block_size=hash_block_size))[1] == snapshot


@pytest.mark.parametrize("dcp", [1, 2])
def test_partial_mamba_state_keeps_upstream_copy_on_write(monkeypatch, dcp):
    manager = _manager(monkeypatch, dcp=dcp, mamba_block=256)
    owner = _request("partial-owner", prompt_length=400)
    blocks, hit, _ = manager.get_computed_blocks(owner)
    assert manager.allocate_slots(owner, 384, hit, blocks) is not None
    manager.free(owner)
    manager.new_step_starts()

    consumer = _request("partial-consumer", prompt_length=400)
    blocks, hit, _ = manager.get_computed_blocks(consumer)
    assert hit == 384
    sources = [group[-1] for group in blocks.blocks]
    source_hashes = [block.block_hash for block in sources]
    new_blocks = manager.allocate_slots(consumer, 16, hit, blocks)
    assert new_blocks is not None
    copies, retained = manager.take_kv_cache_block_copies()
    for source, group in zip(sources, new_blocks.get_block_ids()):
        assert KVCacheBlockCopy(source.block_id, group[0]) in copies
    assert [block.block_hash for block in sources] == source_hashes
    manager.block_pool.free_blocks(retained)
