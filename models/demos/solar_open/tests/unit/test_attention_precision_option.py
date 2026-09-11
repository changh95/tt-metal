# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the attention bf16-output option (phase 2, design_misc.md (a)). No device is opened.

    pytest models/demos/solar_open/tests/unit/test_attention_precision_option.py

``SOLAR_OPEN_ATTENTION_BF16_OUTPUT=1`` (or ``bf16_output=True`` on the attention program config) skips the two bfp8
typecasts of the attention branch: the o_proj input in prefill (``apply_output_projection``) and the pre-all_reduce
partial in decode. The default must stay the phase-1 behaviour (both casts). ``ttnn`` is replaced by a MagicMock in
the operations module so the op sequence can be asserted without a device.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from models.demos.solar_open.tt.attention import operations as ops
from models.demos.solar_open.tt.attention.config import ProgramConfig

ENV = ops.ATTENTION_BF16_OUTPUT_ENV


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


class TestOptionSwitch:
    def test_env_name(self):
        assert ENV == "SOLAR_OPEN_ATTENTION_BF16_OUTPUT"

    def test_default_off(self):
        assert ops.attention_bf16_output() is False
        assert ops.attention_bf16_output(ProgramConfig()) is False  # the base config carries no bf16_output field
        assert ops.attention_bf16_output(None) is False

    @pytest.mark.parametrize("value, expected", [("1", True), ("0", False), ("", False), ("true", False)])
    def test_env(self, monkeypatch, value, expected):
        monkeypatch.setenv(ENV, value)
        assert ops.attention_bf16_output(ProgramConfig()) is expected

    def test_program_config_attribute(self, monkeypatch):
        # A future SolarOpenAttentionProgramConfig.bf16_output field turns the option on without the env ...
        assert ops.attention_bf16_output(SimpleNamespace(bf16_output=True)) is True
        assert ops.attention_bf16_output(SimpleNamespace(bf16_output=False)) is False
        # ... and the env still wins when the field is False (either source suffices)
        monkeypatch.setenv(ENV, "1")
        assert ops.attention_bf16_output(SimpleNamespace(bf16_output=False)) is True


class TestApplyOutputProjection:
    @pytest.fixture
    def mock_ttnn(self, monkeypatch):
        fake = MagicMock(name="ttnn")
        monkeypatch.setattr(ops, "ttnn", fake)
        return fake

    def _weights(self):
        return SimpleNamespace(o_proj=MagicMock(name="o_proj"))

    def test_default_casts_input_to_bfp8(self, mock_ttnn):
        """Phase-1 behaviour: typecast -> matmul -> free the bfp8 copy (the caller frees the bf16 input)."""
        tensor, weights = MagicMock(name="sdpa_out"), self._weights()
        out = ops.apply_output_projection(tensor, weights, mock_ttnn.bfloat16)
        mock_ttnn.typecast.assert_called_once_with(tensor, mock_ttnn.bfloat8_b)
        cast = mock_ttnn.typecast.return_value
        mock_ttnn.matmul.assert_called_once_with(cast, weights.o_proj, dtype=mock_ttnn.bfloat16)
        cast.deallocate.assert_called_once_with(True)
        tensor.deallocate.assert_not_called()
        assert out is mock_ttnn.matmul.return_value

    def test_keep_bf16_skips_the_cast(self, mock_ttnn):
        tensor, weights = MagicMock(name="sdpa_out"), self._weights()
        out = ops.apply_output_projection(tensor, weights, mock_ttnn.bfloat16, keep_bf16=True)
        mock_ttnn.typecast.assert_not_called()
        mock_ttnn.matmul.assert_called_once_with(tensor, weights.o_proj, dtype=mock_ttnn.bfloat16)
        tensor.deallocate.assert_not_called()  # the caller owns the input
        assert out is mock_ttnn.matmul.return_value

    def test_keep_bf16_still_casts_for_bfp8_activations(self, mock_ttnn):
        """Prefill above 32K tokens runs with a bfp8 activation dtype: the input cast stays (DRAM at that length)."""
        tensor, weights = MagicMock(name="sdpa_out"), self._weights()
        ops.apply_output_projection(tensor, weights, mock_ttnn.bfloat8_b, keep_bf16=True)
        mock_ttnn.typecast.assert_called_once_with(tensor, mock_ttnn.bfloat8_b)
        mock_ttnn.matmul.assert_called_once_with(
            mock_ttnn.typecast.return_value, weights.o_proj, dtype=mock_ttnn.bfloat8_b
        )


class TestWiring:
    """The prefill / decode modules consult the same switch (source-level check, no device)."""

    def test_prefill_passes_keep_bf16(self):
        import inspect

        from models.demos.solar_open.tt.attention import prefill

        src = inspect.getsource(prefill.prefill_forward)
        # Since phase 3g / D2 the switch is read once (the packed-pass program configs of packed_numerics need it too)
        # and the value is what the o_proj call receives.
        assert "keep_bf16 = attention_bf16_output(program_config)" in src
        assert "keep_bf16=keep_bf16," in src
        assert "attention_seq_numerics_configs(" in src

    def test_decode_guards_the_typecast(self):
        import inspect

        from models.demos.solar_open.tt.attention import decode

        src = inspect.getsource(decode.decode_forward)
        # Two cast sites since phase 3e / A2 -- the fused all_reduce_async path (SOLAR_OPEN_DECODE_CCL=fused) casts the
        # o_proj partial in its own layout, the composite path casts the interleaved copy -- each behind the guard.
        assert src.count("ttnn.typecast(tt_out, ttnn.bfloat8_b)") == 2
        assert src.count("if not attention_bf16_output(program_config):") == 2
