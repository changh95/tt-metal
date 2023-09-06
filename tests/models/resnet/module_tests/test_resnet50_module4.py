"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from typing import Type, Union, Optional, Callable
from loguru import logger
import torch
from torchvision import models, transforms
import pytest


import tt_lib as ttl
from torch_resnet import _make_layer, Bottleneck
from models.utility_functions import comp_allclose_and_pcc, comp_pcc

batch_size=1

@pytest.mark.parametrize("fuse_ops", [False, True], ids=['Not Fused', "Ops Fused"])
def test_resnet50_module4(fuse_ops, imagenet_sample_input):
    image = imagenet_sample_input
    with torch.no_grad():

        torch_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        torch_resnet.eval()
        state_dict = torch_resnet.state_dict()
        torch_module = torch_resnet.layer4

        layer4 = _make_layer(Bottleneck, 512, 3, name="layer4", stride=2, dilate=False, state_dict=state_dict)
        layer4.eval()

        transformed_input = torch_resnet.conv1(image)
        transformed_input = torch_resnet.bn1(transformed_input)
        transformed_input = torch_resnet.relu(transformed_input)
        transformed_input = torch_resnet.maxpool(transformed_input)
        transformed_input = torch_resnet.layer1(transformed_input)
        transformed_input = torch_resnet.layer2(transformed_input)
        input = torch_resnet.layer3(transformed_input)
        print(layer4)
        if fuse_ops:
            modules_to_fuse = [['0.conv1', '0.bn1', '0.relu1'], ['0.conv2', '0.bn2', '0.relu2'], ['0.conv3', '0.bn3']]
            modules_to_fuse.extend([['1.conv1', '1.bn1', '1.relu1'], ['1.conv2', '1.bn2', '1.relu2'], ['1.conv3', '1.bn3']])
            modules_to_fuse.extend([['2.conv1', '2.bn1', '2.relu1'], ['2.conv2', '2.bn2', '2.relu2'], ['2.conv3', '2.bn3']])
            modules_to_fuse.extend([['0.downsample.0', '0.downsample.1']])
            layer4 = torch.ao.quantization.fuse_modules(layer4, modules_to_fuse)

        torch_output = torch_module(input)
        tt_output = layer4(input)

        passing, info = comp_pcc(torch_output, tt_output)
        # we cannot use comp_allclose_and_pcc because the values are close, rtol ends up being nan.logger.info(f"{passing}, {info}")

        logger.info(f"{passing}, {info}")
        assert passing
