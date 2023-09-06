"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

import torch
import tt_lib

from tests.models.yolov5.tt.yolov5_conv import TtYolov5Conv
from tests.models.yolov5.tt.yolov5_bottleneck import TtYolov5Bottleneck


class TtYolov5C3(torch.nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        state_dict,
        base_address,
        device,
        c1,
        c2,
        n=1,
        shortcut=True,
        g=1,
        e=0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        self.cv1 = TtYolov5Conv(
            state_dict=state_dict,
            base_address=f"{base_address}.cv1",
            device=device,
            c1=c1,
            c2=c_,
            k=1,
            s=1,
        )

        self.cv2 = TtYolov5Conv(
            state_dict=state_dict,
            base_address=f"{base_address}.cv2",
            device=device,
            c1=c1,
            c2=c_,
            k=1,
            s=1,
        )

        self.cv3 = TtYolov5Conv(
            state_dict=state_dict,
            base_address=f"{base_address}.cv3",
            device=device,
            c1=2 * c_,
            c2=c2,
            k=1,
        )

        self.m = torch.nn.Sequential(
            *(
                TtYolov5Bottleneck(
                    state_dict,
                    f"{base_address}.m.{i}",
                    device,
                    c_,
                    c_,
                    shortcut,
                    g,
                    e=1.0,
                )
                for i in range(n)
            )
        )

    def forward(self, x):
        return self.cv3(
            tt_lib.fallback_ops.fallback_ops.concat(
                (self.m(self.cv1(x)), self.cv2(x)), 1
            )
        )
