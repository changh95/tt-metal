# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Motif-Technologies/Motif-3 (314B MoE) port for Tenstorrent Blackhole Galaxy.

The ``reference`` subpackage is a pure-PyTorch, device-free golden model. It never imports ``ttnn``.
The ``tt`` subpackage holds the ttnn implementation; see ``README.md`` (CONVENTIONS) for the module contract.

Import rule: nothing under ``models/demos/motif3`` imports another ``models/demos/**`` package at module import
time (vLLM imports the bridge before the mesh opens; demo imports can open the cluster). This file imports nothing.
"""
