"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

from transformers import ViTForImageClassification as HF_ViTForImageClassication
from loguru import logger
import torch

import tt_lib

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)

from models.utility_functions import comp_pcc, comp_allclose_and_pcc

from models.vit.tt.modeling_vit import TtViTIntermediate


def test_vit_intermediate(pcc=0.99):
    hidden_state_shape = (1, 1, 197, 768)

    hidden_state = torch.randn(hidden_state_shape)
    with torch.no_grad():
        HF_model = HF_ViTForImageClassication.from_pretrained(
            "google/vit-base-patch16-224"
        )

        state_dict = HF_model.state_dict()
        reference = HF_model.vit.encoder.layer[5].intermediate

        config = HF_model.config
        HF_output = reference(hidden_state.squeeze(0))

        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)


        tt_hidden_state = torch_to_tt_tensor_rm(
            hidden_state, device, put_on_device=False
        )
        tt_layer = TtViTIntermediate(
            config,
            base_address="vit.encoder.layer.5.intermediate",
            state_dict=state_dict,
            device=device,
        )

        tt_output = tt_layer(tt_hidden_state)
        tt_output = tt_to_torch_tensor(tt_output).squeeze(0)
        pcc_passing, _ = comp_pcc(HF_output, tt_output, pcc)
        _, pcc_output = comp_allclose_and_pcc(HF_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        tt_lib.device.CloseDevice(device)
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."
