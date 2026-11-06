"""Small CPU regressions exercising real predictor/loader/embedding-sharing code."""

import json
import sys
import types
from unittest.mock import patch

import torch
from torch import nn
from vllm.model_executor.models import bailing_moe_v3_mtp as m
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.v1.worker.gpu.spec_decode.eagle import utils


class Embed(nn.Embedding):
    def __init__(self, n, d, **kw):
        super().__init__(n, d)


class Norm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))


class Layer(nn.Module):
    def __init__(self, *, vllm_config, **kw):
        super().__init__()
        self.enorm = Norm(4)

    def forward(self, input_ids, positions, previous_hidden_states, inputs_embeds, *args):
        assert inputs_embeds.is_floating_point(), "Token IDs reached the MTP floating-point path"
        assert inputs_embeds.shape == previous_hidden_states.shape
        return inputs_embeds + previous_hidden_states


cfg = types.SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1, vocab_size=8, hidden_size=4, num_experts=1)
vc = types.SimpleNamespace(speculative_config=None, model_config=types.SimpleNamespace(hf_config=cfg))
weights = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32


def model():
    inner = m.BailingMoeV3MultiTokenPredictor(vllm_config=vc, prefix="model")
    obj = m.BailingMoeV3MTPModel.__new__(m.BailingMoeV3MTPModel)
    nn.Module.__init__(obj)
    obj.model = inner
    obj.config = cfg
    obj.quant_config = None
    obj.get_expert_mapping = lambda: []
    obj.hf_to_vllm_mapper = types.SimpleNamespace(apply=lambda x: x)
    return obj


def load(obj, embed=True):
    ws = [("model.layers.2.enorm.weight", torch.ones(4))]
    if embed:
        ws.insert(0, ("model.word_embeddings.weight", weights.clone()))
    return obj.load_weights(ws)


def checks():
    out = []
    group = types.SimpleNamespace(is_first_rank=False, is_last_rank=True, world_size=2)
    with (
        patch.object(m, "get_pp_group", return_value=group),
        patch.object(utils, "get_pp_group", return_value=group),
        patch.object(m, "VocabParallelEmbedding", Embed),
        patch.object(m, "BailingMoeV3MultiTokenPredictorLayer", Layer),
        patch.object(m, "LogitsProcessor", lambda n: None),
    ):
        obj = model()
        ids = torch.tensor([1, 3], dtype=torch.int32)
        embedded = obj.model.embed_input_ids(ids)
        assert embedded.is_floating_point() and embedded.shape == (2, 4), (
            "PP last-stage drafter embedding is a pass-through"
        )
        loaded = load(obj)
        assert "model.embed_tokens.weight" in loaded and obj.has_own_embed_tokens
        assert torch.equal(obj.model.embed_input_ids(ids), weights[ids.long()])
        result = obj.model(ids, torch.arange(2), torch.zeros(2, 4))
        assert torch.equal(result, weights[ids.long()])
        out.append("PP2-last-embedding-and-checkpoint")
        old = obj.model.embed_tokens
        utils.maybe_share_target_embed(obj, obj.model, types.SimpleNamespace(embed_tokens=PPMissingLayer()))
        assert obj.model.embed_tokens is old
        out.append("PP2-target-placeholder-not-shared")
        try:
            load(model(), False)
        except ValueError as e:
            assert "requires input embedding" in str(e)
        else:
            raise AssertionError("PP2 accepted missing checkpoint embedding")
        out.append("PP2-missing-embedding-fails-closed")
        group.is_first_rank = True
        group.world_size = 1
        obj = model()
        load(obj)
        target = Embed(8, 4)
        target.weight.data.copy_(weights)
        utils.maybe_share_target_embed(obj, obj.model, types.SimpleNamespace(embed_tokens=target))
        assert obj.model.embed_tokens is target
        out.append("PP1-matching-embedding-sharing")
        obj = model()
        load(obj, False)
        assert not obj.has_own_embed_tokens
        utils.maybe_share_target_embed(obj, obj.model, types.SimpleNamespace(embed_tokens=target))
        assert obj.model.embed_tokens is target
        out.append("PP1-checkpoint-without-embedding-fallback")
    return out


if "--expect-broken" in sys.argv:
    try:
        checks()
    except AssertionError as e:
        assert "PP last-stage drafter embedding is a pass-through" in str(e), str(e)
        print("PREPATCH_REPRODUCED " + str(e))
    else:
        raise AssertionError("Pre-patch failure was not reproduced")
else:
    print("REGRESSION_PASS " + json.dumps(checks()))
