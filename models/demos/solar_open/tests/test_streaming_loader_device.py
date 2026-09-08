# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device smoke of the phase-2 streaming loader (DESIGN.md 4.16, design_loader.md 5.2).

Builds a 2-layer Solar-Open model through ``create_tt_model`` with ``SOLAR_OPEN_STREAMING_LOAD=1`` into a TEMPORARY
``TT_CACHE_PATH`` (so a cold host load runs, per access over the safetensors shards instead of the 393 GB
``from_pretrained``), then asserts

1. the state dict handed back is the ``LazyStateDict`` with no repeat reads (every tensor read once), 2 layers touched
   and 4 fused expert builds (gate_up + down for both layers);
2. every tensorbin the build wrote (embedding, lm_head, final norm, ``model.layers.{0,1}/**``) is byte-identical to the
   same file of the phase-1 cache built from the whole-model load (``$TT_CACHE_PATH/tensor_cache_<...>_(1, 8)``): the
   device weights of a streamed build are the phase-1 weights, so the teacher-forced numbers carry over;
3. the ``.weights_complete`` marker is NOT written (a ``num_layers``-limited build is partial);
4. the peak host RSS stays below 40 GB (phase 1: 393 GB);
5. one decode step of the 2-layer model runs (the device tensors are usable).

    SOLAR_OPEN_STREAMING_LOAD=1 timeout 1800 pytest models/demos/solar_open/tests/test_streaming_loader_device.py \
        -k 1x8 -x -p no:cacheprovider

The temporary cache (~7 GB next to ``$TT_CACHE_PATH``) is deleted afterwards unless
``SOLAR_OPEN_KEEP_STREAM_SMOKE_CACHE=1``. Needs the full checkpoint (``HF_MODEL``) and the phase-1 cache for the byte
comparison (skips otherwise).
"""

import filecmp
import os
import resource
import shutil
import tempfile
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.solar_open.tt.common import create_tt_model
from models.demos.solar_open.tt.model_config import ModelArgs
from models.demos.solar_open.utils.streaming_loader import LazyStateDict

from .test_factory import parametrize_mesh_with_fabric

NUM_LAYERS = 2
MAX_RSS_GB = 40.0


def _phase1_cache_dir(cache_root, cache_dir_name):
    ref = Path(cache_root) / cache_dir_name
    if not (ref / ModelArgs.WEIGHT_CACHE_MARKER).is_file():
        pytest.skip(
            f"phase-1 reference cache {ref} is not complete (no {ModelArgs.WEIGHT_CACHE_MARKER}); build it first"
        )
    return ref


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # ru_maxrss is KiB on Linux


@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric([(1, 8)])
def test_streaming_loader_device_smoke(mesh_device, device_params, monkeypatch):
    model_path = os.getenv("HF_MODEL")
    if not model_path or not os.path.isdir(model_path):
        pytest.skip("HF_MODEL must point at the local Solar-Open-100B checkpoint directory")
    if not (Path(model_path) / "model.safetensors.index.json").is_file():
        pytest.skip("the checkpoint index (model.safetensors.index.json) is required for the streaming loader")
    cache_root = os.getenv("TT_CACHE_PATH") or model_path
    dtype = ttnn.bfloat8_b

    tmp_root = tempfile.mkdtemp(prefix="tt_cache_stream_smoke.", dir=str(Path(cache_root).parent))
    monkeypatch.setenv("TT_CACHE_PATH", tmp_root)
    monkeypatch.setenv("SOLAR_OPEN_STREAMING_LOAD", "1")
    monkeypatch.delenv("SOLAR_OPEN_FORCE_MODEL_LOAD", raising=False)
    rss_before = _rss_gb()
    state_dict = None
    try:
        t0 = time.perf_counter()
        model_args, model, tt_kv_cache, state_dict = create_tt_model(
            mesh_device,
            max_batch_size=1,
            max_seq_len=4096,
            paged_attention_config=None,
            dtype=dtype,
            num_layers=NUM_LAYERS,
        )
        build_s = time.perf_counter() - t0
        rss_after = _rss_gb()

        # 1. the loader itself came back, walked exactly once
        assert isinstance(state_dict, LazyStateDict), type(state_dict)
        stats = dict(state_dict.stats)
        logger.info(
            f"streamed {NUM_LAYERS}-layer build: {build_s:.1f} s wall, loader {stats}, "
            f"ru_maxrss {rss_before:.2f} -> {rss_after:.2f} GB"
        )
        assert stats["repeat_reads"] == 0, stats
        assert stats["layers_touched"] == NUM_LAYERS, stats
        assert stats["fused_builds"] == 2 * NUM_LAYERS, stats
        assert stats["bytes"] > 8 * 2**30, stats  # 2 layers (2 x 4.2 GB) + embedding + lm_head + norm

        # 2. byte-identical tensorbins against the phase-1 (whole-model load) cache
        cache_dir = model_args.weight_cache_path(dtype)
        assert Path(tmp_root) in cache_dir.parents, cache_dir
        reference_dir = _phase1_cache_dir(cache_root, cache_dir.name)
        written = sorted(p for p in cache_dir.rglob("*.tensorbin"))
        assert written, f"no tensorbin written under {cache_dir}"
        layers_written = {
            p.relative_to(cache_dir).parts[0]
            for p in written
            if p.relative_to(cache_dir).parts[0].startswith("model.layers.")
        }
        assert layers_written == {f"model.layers.{i}" for i in range(NUM_LAYERS)}, layers_written
        differing, missing, total_bytes = [], [], 0
        for path in written:
            rel = path.relative_to(cache_dir)
            ref = reference_dir / rel
            if not ref.is_file():
                missing.append(str(rel))
                continue
            total_bytes += path.stat().st_size
            if not filecmp.cmp(path, ref, shallow=False):
                differing.append(str(rel))
        logger.info(
            f"{len(written)} tensorbins ({total_bytes / 2**30:.2f} GiB) compared byte-for-byte with {reference_dir}: "
            f"{len(differing)} differ, {len(missing)} missing in the reference"
        )
        assert not missing, f"tensorbins without a phase-1 counterpart: {missing}"
        assert not differing, f"tensorbins differing from the phase-1 cache: {differing}"

        # 3. partial build: no completion marker
        assert not (
            cache_dir / ModelArgs.WEIGHT_CACHE_MARKER
        ).exists(), "a num_layers-limited build must not mark the cache complete"

        # 4. host memory
        assert rss_after < MAX_RSS_GB, f"peak host RSS {rss_after:.1f} GB (phase-1 whole-model load: 393 GB)"

        # 5. one decode step (2-layer garbage logits, must be finite)
        tokens = torch.tensor([1234], dtype=torch.long)
        current_pos = torch.zeros(1, dtype=torch.long)
        tt_tokens, tt_pos, tt_rope_idxs, _ = model.prepare_inputs_decode(tokens, current_pos, page_table=None)
        tt_logits, _ = model.ttnn_decode_forward(
            tokens=tt_tokens, current_pos=tt_pos, rot_mat_idxs=tt_rope_idxs, page_table=None, kv_cache=None
        )
        logits = ttnn.to_torch(
            tt_logits,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=tuple(mesh_device.shape)),
        )
        logits = logits.reshape(-1, logits.shape[-1])[:1, : model_args.vocab_size].float()
        assert torch.isfinite(logits).all(), "non-finite logits from the streamed 2-layer model"
        logger.info(f"decode step ok: logits {tuple(logits.shape)} argmax {int(logits.argmax(-1))}")
    finally:
        if state_dict is not None and hasattr(state_dict, "close"):
            state_dict.close()
        if os.getenv("SOLAR_OPEN_KEEP_STREAM_SMOKE_CACHE") == "1":
            logger.info(f"keeping the smoke cache at {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)
