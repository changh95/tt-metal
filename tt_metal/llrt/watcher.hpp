/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tt_metal/impl/device/device.hpp"

namespace tt {
namespace llrt {

void watcher_init(tt::tt_metal::Device *dev);
void watcher_attach(tt::tt_metal::Device *dev);
void watcher_detach(tt::tt_metal::Device *dev);

} // namespace llrt
} // namespace tt
