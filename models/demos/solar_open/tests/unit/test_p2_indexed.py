# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the perf-p2 indexed/gather single-user decode path (no device).

* ``MoEOptions.indexed_decode``: default on, ``SOLAR_OPEN_INDEXED_DECODE=0`` off, cache-neutral (not in the marker).
* ``mlp.indexed_decode_enabled``: the enabling rule (flag, unfused shared expert, EP=1, fused router).
* ``experts.IndexedRouting``: layout / dtype / shape validation of the contract the sparse_matmul indexed mode needs.
* ``decode_forward`` argument guards (no device tensors touched before the guards fire).
* Source-level contracts of ``_decode_forward_indexed`` / ``TopKRouter.route_indexed`` that the device tests cannot
  observe: both sparse_matmuls take ``indices=`` and never ``nnz=`` (the op rejects the pair), the down uses the
  compact A (``is_input_a_sparse=True``), the routing-weight mul sits on the down input or on the down output
  (``INDEXED_WEIGHTS_ON_DOWN_INPUT``), and the indexed router skips the dense scatter.

    SOLAR_OPEN_NUM_DEVICES=8 pytest models/demos/solar_open/tests/unit/test_p2_indexed.py
"""

import inspect
import re
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.solar_open.config import MoEOptions
from models.demos.solar_open.tt import mlp as mlp_module
from models.demos.solar_open.tt import topk as topk_module
from models.demos.solar_open.tt.experts import decode as decode_module
from models.demos.solar_open.tt.experts.config import IndexedRouting


class TestMoEOptionsIndexed:
    def test_default_on_and_env_off(self, monkeypatch):
        assert MoEOptions().indexed_decode is True
        monkeypatch.delenv("SOLAR_OPEN_INDEXED_DECODE", raising=False)
        assert MoEOptions.from_env().indexed_decode is True
        monkeypatch.setenv("SOLAR_OPEN_INDEXED_DECODE", "0")
        assert MoEOptions.from_env().indexed_decode is False
        monkeypatch.setenv("SOLAR_OPEN_INDEXED_DECODE", "1")
        assert MoEOptions.from_env().indexed_decode is True

    def test_cache_neutral(self):
        on, off = MoEOptions(), MoEOptions(indexed_decode=False)
        assert on != off
        assert on.marker_fields() == off.marker_fields()
        assert "indexed_decode" not in on.marker_fields()
        assert on.expert_dtype_str == off.expert_dtype_str


class TestEnablingRule:
    @pytest.mark.parametrize(
        "options, fuse_shared, ep, expected, reason_fragment",
        [
            (MoEOptions(), False, 1, True, ""),
            (MoEOptions(indexed_decode=False), False, 1, False, "SOLAR_OPEN_INDEXED_DECODE=0"),
            (MoEOptions(), True, 1, False, "fused shared expert"),
            (MoEOptions(), False, 2, False, "EP=2"),
            (MoEOptions(router_impl="ops"), False, 1, False, "uint32"),
            (MoEOptions(router_fp32_logits=False), False, 1, True, ""),  # bf16 logits still give uint16 ids
        ],
    )
    def test_rule(self, options, fuse_shared, ep, expected, reason_fragment):
        enabled, why = mlp_module.indexed_decode_enabled(options, fuse_shared, ep)
        assert enabled is expected
        assert reason_fragment in why
        if expected:
            assert why == ""


def _fake_tensor(shape, dtype, layout):
    return SimpleNamespace(shape=shape, dtype=dtype, layout=layout, deallocate=lambda force=True: None)


class TestIndexedRouting:
    def test_accepts_contract(self):
        r = IndexedRouting(
            indices=_fake_tensor((1, 1, 1, 8), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT),
            weights=_fake_tensor((1, 1, 1, 8), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            top_k=8,
        )
        assert r.top_k == 8
        r.deallocate()  # both fakes accept the call

    @pytest.mark.parametrize(
        "indices, weights, top_k, fragment",
        [
            ((1, 8), ttnn.uint16, 8, "must be (1, 1, 1, 8)"),
            ((1, 1, 1, 8), ttnn.uint32, 8, "uint16 ROW_MAJOR"),
            ((1, 1, 1, 8), ttnn.uint16, 0, "top_k must be positive"),
            ((1, 1, 1, 4), ttnn.uint16, 8, "must be (1, 1, 1, 8)"),
        ],
    )
    def test_rejects(self, expect_error, indices, weights, top_k, fragment):
        with expect_error(ValueError, re.escape(fragment)):
            IndexedRouting(
                indices=_fake_tensor(indices, weights, ttnn.ROW_MAJOR_LAYOUT),
                weights=_fake_tensor((1, 1, 1, 8), ttnn.bfloat16, ttnn.TILE_LAYOUT),
                top_k=top_k,
            )

    def test_rejects_indices_in_tile_layout(self, expect_error):
        with expect_error(ValueError, "uint16 ROW_MAJOR"):
            IndexedRouting(
                indices=_fake_tensor((1, 1, 1, 8), ttnn.uint16, ttnn.TILE_LAYOUT),
                weights=_fake_tensor((1, 1, 1, 8), ttnn.bfloat16, ttnn.TILE_LAYOUT),
                top_k=8,
            )

    def test_rejects_row_major_weights(self, expect_error):
        with expect_error(ValueError, "bfloat16 TILE"):
            IndexedRouting(
                indices=_fake_tensor((1, 1, 1, 8), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT),
                weights=_fake_tensor((1, 1, 1, 8), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                top_k=8,
            )


def _routing():
    return IndexedRouting(
        indices=_fake_tensor((1, 1, 1, 8), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT),
        weights=_fake_tensor((1, 1, 1, 8), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        top_k=8,
    )


class TestDecodeForwardGuards:
    """The guards fire on the shapes / flags alone, before any device op."""

    def _call(self, seq_len, always_on=0, **kwargs):
        hidden = SimpleNamespace(shape=(1, 1, seq_len, 4096))
        weights = SimpleNamespace(num_always_on_experts=always_on)
        return decode_module.decode_forward(
            hidden, kwargs.pop("routing_weights", None), weights, None, None, None, None, None, **kwargs
        )

    def test_indexed_needs_single_user(self, expect_error):
        with expect_error(ValueError, "exactly one token"):
            self._call(2, indexed_routing=_routing(), sparsity_placeholder=object())

    def test_indexed_rejects_always_on_slots(self, expect_error):
        with expect_error(ValueError, "always-on"):
            self._call(1, always_on=1, indexed_routing=_routing(), sparsity_placeholder=object())

    def test_indexed_needs_placeholder(self, expect_error):
        with expect_error(ValueError, "sparsity_placeholder"):
            self._call(1, indexed_routing=_routing())

    def test_dense_path_needs_routing_weights(self, expect_error):
        with expect_error(ValueError, "routing_weights is required"):
            self._call(1)

    def test_unknown_weights_layout_mode(self, expect_error):
        with expect_error(ValueError, "INDEXED_WEIGHTS_LAYOUT"):
            decode_module._indexed_expert_scalars(object(), 8, mode="bogus")

    def test_shipped_layout_mode_is_known(self):
        assert decode_module.INDEXED_WEIGHTS_LAYOUT in ("transpose", "row_major")
        assert isinstance(decode_module.INDEXED_WEIGHTS_ON_DOWN_INPUT, bool)


class TestSourceContracts:
    def test_indexed_path_calls(self):
        src = inspect.getsource(decode_module._decode_forward_indexed)
        calls = [m.start() for m in re.finditer(r"ttnn\.sparse_matmul\(", src)]
        assert len(calls) == 2, "gate|up and down: exactly two sparse_matmuls"
        assert src.count("indices=indexed_routing.indices") == 2, "both sparse_matmuls must take the top-k ids"
        assert "nnz=" not in src, "nnz must not be passed together with indices (the op rejects it)"
        assert "is_input_a_sparse=True" in src and "is_input_b_sparse=True" in src, "compact A for the down"
        reduce_at = src.index("fast_reduce_nc(down, dims=[1]")
        assert reduce_at > calls[1]
        # both weight placements exist and sit where they must: on the down input (between the two sparse_matmuls,
        # INDEXED_WEIGHTS_ON_DOWN_INPUT) or on the down output (after the down, before the k-slot reduction)
        mul_in_at = src.index("ttnn.mul(down_input, expert_scalars")
        assert calls[0] < mul_in_at < calls[1], "the down-input mul must sit between gate|up and the down"
        mul_out_at = src.index("ttnn.mul(down, expert_scalars")
        assert calls[1] < mul_out_at < reduce_at, "the down-output mul must sit between the down and the reduction"
        assert src.index("if INDEXED_WEIGHTS_ON_DOWN_INPUT:") < mul_in_at
        assert src.index("if not INDEXED_WEIGHTS_ON_DOWN_INPUT:") < mul_out_at
        assert (
            src.index("shared_expert(hidden_states)") < calls[0]
        ), "the shared partial is computed before the input is consumed"
        assert src.index("ttnn.add(next_states, shared") < src.index("apply_tensor_parallel_allreduce(")

    def test_route_indexed_skips_the_scatter(self):
        src = inspect.getsource(topk_module.TopKRouter.route_indexed)
        assert "ttnn.scatter(" not in src  # the docstring names the op; the body must not call it
        assert "ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)" in src
        call_src = inspect.getsource(topk_module.TopKRouter.__call__)
        assert "ttnn.scatter(" in call_src  # the dense form is unchanged

    def test_mlp_route_is_the_single_dispatch(self):
        src = inspect.getsource(mlp_module.MLP.route)
        assert "route_indexed" in src and "self.indexed_decode" in src
        call_src = inspect.getsource(mlp_module.MLP.__call__)
        assert "self.route(" in call_src and "indexed_routing=indexed" in call_src
