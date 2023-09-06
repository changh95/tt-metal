// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_soc_descriptor.h"

#include <iostream>
#include <string>
#include <fstream>
#include "yaml-cpp/yaml.h"
#include "device/tt_device.h"
#include "dev_mem_map.h"

#include "common/assert.hpp"

bool tt_SocDescriptor::is_worker_core(const CoreCoord &core) const {
    return (
        routing_x_to_worker_x.find(core.x) != routing_x_to_worker_x.end() &&
        routing_y_to_worker_y.find(core.y) != routing_y_to_worker_y.end());
}
CoreCoord tt_SocDescriptor::get_worker_core(const CoreCoord &core) const {
    CoreCoord worker_xy = {
        static_cast<size_t>(routing_x_to_worker_x.at(core.x)), static_cast<size_t>(routing_y_to_worker_y.at(core.y))};
    return worker_xy;
}


int tt_SocDescriptor::get_num_dram_channels() const {
    int num_channels = 0;
    for (const auto& dram_core : dram_cores) {
        if (dram_core.size() > 0) {
            num_channels++;
        }
    }
    return num_channels;
}
CoreCoord tt_SocDescriptor::get_core_for_dram_channel(int dram_chan, int subchannel) const {
    tt::log_assert(dram_chan < this->dram_cores.size(), "dram_chan={} must be within range of num_dram_channels={}", dram_chan, this->dram_cores.size());
    tt::log_assert(subchannel < this->dram_cores.at(dram_chan).size(), "subchannel={} must be within range of num_subchannels={}", subchannel, this->dram_cores.at(dram_chan).size());
    return this->dram_cores.at(dram_chan).at(subchannel);
};
CoreCoord tt_SocDescriptor::get_preferred_worker_core_for_dram_channel(int dram_chan) const {
    tt::log_assert(dram_chan < this->preferred_worker_dram_core.size(), "dram_chan={} must be within range of preferred_worker_dram_core.size={}", dram_chan, this->preferred_worker_dram_core.size());
    return this->preferred_worker_dram_core.at(dram_chan);
};
CoreCoord tt_SocDescriptor::get_preferred_eth_core_for_dram_channel(int dram_chan) const {
    tt::log_assert(dram_chan < this->preferred_eth_dram_core.size(), "dram_chan={} must be within range of preferred_eth_dram_core.size={}", dram_chan, this->preferred_eth_dram_core.size());
    return this->preferred_eth_dram_core.at(dram_chan);
};
size_t tt_SocDescriptor::get_address_offset(int dram_chan) const {
    tt::log_assert(dram_chan < this->dram_address_offsets.size(), "dram_chan={} must be within range of dram_address_offsets.size={}", dram_chan, this->dram_address_offsets.size());
    return this->dram_address_offsets.at(dram_chan);
}
int tt_SocDescriptor::get_num_dram_subchans() const {
    int num_chan = 0;
    for (const std::vector<CoreCoord> &core : this->dram_cores) {
        num_chan += core.size();
    }
    return num_chan;
}
int tt_SocDescriptor::get_num_dram_blocks_per_channel() const {
    int num_blocks = 0;
    if (arch == tt::ARCH::GRAYSKULL) {
        num_blocks = 1;
    } else if (arch == tt::ARCH::WORMHOLE) {
        num_blocks = 2;
    } else if (arch == tt::ARCH::WORMHOLE_B0) {
        num_blocks = 2;
    }
    return num_blocks;
}


bool tt_SocDescriptor::is_ethernet_core(const CoreCoord &core) const {
    return this->ethernet_core_channel_map.find(core) != ethernet_core_channel_map.end();
}

bool tt_SocDescriptor::is_harvested_core(const CoreCoord &core) const {
    for (const auto& core_it : harvested_workers) {
        if (core_it == core) {
            return true;
        }
    }
    return false;
}

const std::vector<CoreCoord>& tt_SocDescriptor::get_pcie_cores() const {
    return pcie_cores;
}

const std::vector<CoreCoord> tt_SocDescriptor::get_dram_cores() const {
    std::vector<CoreCoord> cores;

    // This is inefficient, but is currently not used in a perf path
    for (const auto& channel_it : dram_cores) {
        for (const auto& core_it : channel_it) {
            cores.push_back(core_it);
        }
    }

    return cores;
}

const std::vector<CoreCoord>& tt_SocDescriptor::get_ethernet_cores() const {
    return ethernet_cores;
}

bool tt_SocDescriptor::get_channel_of_ethernet_core(const CoreCoord &core) const {
    return this->ethernet_core_channel_map.at(core);
}

const char* ws = " \t\n\r\f\v";

// trim from end of string (right)
inline std::string& rtrim(std::string& s, const char* t = ws)
{
    s.erase(s.find_last_not_of(t) + 1);
    return s;
}

// trim from beginning of string (left)
inline std::string& ltrim(std::string& s, const char* t = ws)
{
    s.erase(0, s.find_first_not_of(t));
    return s;
}

// trim from both ends of string (right then left)
inline std::string& trim(std::string& s, const char* t = ws)
{
    return ltrim(rtrim(s, t), t);
}

void load_core_descriptors_from_device_descriptor(YAML::Node device_descriptor_yaml, tt_SocDescriptor &soc_descriptor) {
  auto worker_l1_size = device_descriptor_yaml["worker_l1_size"].as<int>();
  auto eth_l1_size = device_descriptor_yaml["eth_l1_size"].as<int>();

  for (const auto &core_node : device_descriptor_yaml["arc"]) {
    CoreDescriptor core_descriptor;
    if (core_node.IsScalar()) {
      core_descriptor.coord = format_node(core_node.as<std::string>());
    } else {
      tt::log_fatal ("Only NOC coords supported for arc cores");
    }
    core_descriptor.type = CoreType::ARC;
    soc_descriptor.cores.insert({core_descriptor.coord, core_descriptor});
    soc_descriptor.arc_cores.push_back(core_descriptor.coord);
  }

  for (const auto &core_node :  device_descriptor_yaml["pcie"]) {
    CoreDescriptor core_descriptor;
    if (core_node.IsScalar()) {
      core_descriptor.coord = format_node(core_node.as<std::string>());
    } else {
      tt::log_fatal ("Only NOC coords supported for pcie cores");
    }
    core_descriptor.type = CoreType::PCIE;
    soc_descriptor.cores.insert({core_descriptor.coord, core_descriptor});
    soc_descriptor.pcie_cores.push_back(core_descriptor.coord);
  }

  int current_dram_channel = 0;
  for (const auto &channel_it : device_descriptor_yaml["dram"]) {
    soc_descriptor.dram_cores.push_back({});
    auto &soc_dram_cores = soc_descriptor.dram_cores.at(soc_descriptor.dram_cores.size() - 1);
    for (int i = 0; i < channel_it.size(); i++) {
      const auto &dram_core = channel_it[i];
      CoreDescriptor core_descriptor;
      if (dram_core.IsScalar()) {
        core_descriptor.coord = format_node(dram_core.as<std::string>());
      } else {
        tt::log_fatal ("Only NOC coords supported for dram cores");
      }
      core_descriptor.type = CoreType::DRAM;
      soc_descriptor.cores.insert({core_descriptor.coord, core_descriptor});
      soc_dram_cores.push_back(core_descriptor.coord);
      soc_descriptor.dram_core_channel_map[core_descriptor.coord] = {current_dram_channel, i};
    }
    current_dram_channel++;
  }

  soc_descriptor.preferred_eth_dram_core.clear();
  for (const auto& core_node: device_descriptor_yaml["dram_preferred_eth_endpoint"]) {
    if (core_node.IsScalar()) {
      soc_descriptor.preferred_eth_dram_core.push_back(format_node(core_node.as<std::string>()));
    } else {
      tt::log_fatal ("Only NOC coords supported for dram_preferred_eth_endpoint cores");
    }
  }
  soc_descriptor.preferred_worker_dram_core.clear();
  for (const auto& core_node: device_descriptor_yaml["dram_preferred_worker_endpoint"]) {
    if (core_node.IsScalar()) {
      soc_descriptor.preferred_worker_dram_core.push_back(format_node(core_node.as<std::string>()));
    } else {
      tt::log_fatal ("Only NOC coords supported for dram_preferred_worker_endpoint");
    }
  }
  soc_descriptor.dram_address_offsets = device_descriptor_yaml["dram_address_offsets"].as<std::vector<size_t>>();

  auto eth_cores = device_descriptor_yaml["eth"].as<std::vector<std::string>>();
  int current_ethernet_channel = 0;
  for (const auto &core_node : device_descriptor_yaml["eth"]) {
    CoreDescriptor core_descriptor;
    if (core_node.IsScalar()) {
      core_descriptor.coord = format_node(core_node.as<std::string>());
    } else {
      tt::log_fatal ("Only NOC coords supported for eth cores");
    }
    core_descriptor.type = CoreType::ETH;
    core_descriptor.l1_size = eth_l1_size;
    soc_descriptor.cores.insert({core_descriptor.coord, core_descriptor});
    soc_descriptor.ethernet_cores.push_back(core_descriptor.coord);

    soc_descriptor.ethernet_core_channel_map[core_descriptor.coord] = current_ethernet_channel;
    current_ethernet_channel++;
  }

  std::set<int> worker_routing_coords_x;
  std::set<int> worker_routing_coords_y;
  std::unordered_map<int,int> routing_coord_worker_x;
  std::unordered_map<int,int> routing_coord_worker_y;
  for (const auto &core_node : device_descriptor_yaml["functional_workers"]) {
    CoreDescriptor core_descriptor;
    if (core_node.IsScalar()) {
      core_descriptor.coord = format_node(core_node.as<std::string>());
    } else {
      tt::log_fatal ("Only NOC coords supported for functional_workers cores");
    }
    core_descriptor.type = CoreType::WORKER;
    core_descriptor.l1_size = worker_l1_size;
    core_descriptor.dram_size_per_core = DEFAULT_DRAM_SIZE_PER_CORE;
    soc_descriptor.cores.insert({core_descriptor.coord, core_descriptor});
    soc_descriptor.workers.push_back(core_descriptor.coord);
    worker_routing_coords_x.insert(core_descriptor.coord.x);
    worker_routing_coords_y.insert(core_descriptor.coord.y);
  }

  int func_x_start = 0;
  int func_y_start = 0;
  std::set<int>::iterator it;
  for (it=worker_routing_coords_x.begin(); it!=worker_routing_coords_x.end(); ++it) {
    soc_descriptor.worker_log_to_routing_x[func_x_start] = *it;
    soc_descriptor.routing_x_to_worker_x[*it] = func_x_start;
    func_x_start++;
  }
  for (it=worker_routing_coords_y.begin(); it!=worker_routing_coords_y.end(); ++it) {
    soc_descriptor.worker_log_to_routing_y[func_y_start] = *it;
    soc_descriptor.routing_y_to_worker_y[*it] = func_y_start;
    func_y_start++;
  }

  soc_descriptor.worker_grid_size = CoreCoord(func_x_start, func_y_start);

  for (const auto &core_node : device_descriptor_yaml["harvested_workers"]) {
    CoreDescriptor core_descriptor;
    if (core_node.IsScalar()) {
      core_descriptor.coord = format_node(core_node.as<std::string>());
    } else {
      tt::log_fatal ("Only NOC coords supported for harvested_workers cores");
    }
    core_descriptor.type = CoreType::HARVESTED;
    soc_descriptor.cores.insert({core_descriptor.coord, core_descriptor});
  }

  for (const auto &core_node :  device_descriptor_yaml["router_only"]) {
    CoreDescriptor core_descriptor;
    if (core_node.IsScalar()) {
      core_descriptor.coord = format_node(core_node.as<std::string>());
    } else {
      tt::log_fatal ("Only NOC/logical coords supported for router_only cores");
    }
    core_descriptor.type = CoreType::ROUTER_ONLY;
    soc_descriptor.cores.insert({core_descriptor.coord, core_descriptor});
  }
}

void load_soc_features_from_device_descriptor(YAML::Node &device_descriptor_yaml, tt_SocDescriptor *soc_descriptor) {
  soc_descriptor->overlay_version = device_descriptor_yaml["features"]["overlay"]["version"].as<int>();
  soc_descriptor->packer_version = device_descriptor_yaml["features"]["packer"]["version"].as<int>();
  soc_descriptor->unpacker_version = device_descriptor_yaml["features"]["unpacker"]["version"].as<int>();
  soc_descriptor->dst_size_alignment = device_descriptor_yaml["features"]["math"]["dst_size_alignment"].as<int>();
  soc_descriptor->worker_l1_size = device_descriptor_yaml["worker_l1_size"].as<int>();
  soc_descriptor->eth_l1_size = device_descriptor_yaml["eth_l1_size"].as<int>();
  soc_descriptor->dram_bank_size = device_descriptor_yaml["dram_bank_size"].as<uint32_t>();
}

// Determines which core will write perf-events on which dram-bank.
// Creates a map of dram cores to worker cores, in the order that they will get dumped.
void map_workers_to_dram_banks(tt_SocDescriptor *soc_descriptor) {
  for (CoreCoord worker:soc_descriptor->workers) {
    TT_ASSERT(soc_descriptor->dram_cores.size() > 0, "No DRAM channels detected");
    // Initialize target dram core to the first dram.
    CoreCoord target_dram_bank = soc_descriptor->dram_cores.at(0).at(0);
    std::vector<std::vector<CoreCoord>> dram_cores_per_channel;
    if (soc_descriptor->arch == tt::ARCH::WORMHOLE || soc_descriptor->arch == tt::ARCH::WORMHOLE_B0) {
      dram_cores_per_channel = {{CoreCoord(0, 0)}, {CoreCoord(0, 5)}, {CoreCoord(5, 0)}, {CoreCoord(5, 2)}, {CoreCoord(5, 3)}, {CoreCoord(5, 5)}};
    } else {
      dram_cores_per_channel = soc_descriptor->dram_cores;
    }
    for (const auto &dram_cores : dram_cores_per_channel) {
      for (CoreCoord dram: dram_cores) {
        int diff_x = worker.x - dram.x;
        int diff_y = worker.y - dram.y;
        // Represents a dram core that comes "before" this worker.
        if (diff_x >= 0 && diff_y >= 0) {
          int diff_dram_x = worker.x - target_dram_bank.x;
          int diff_dram_y = worker.y - target_dram_bank.y;
          // If initial dram core comes after the worker, swap it with this dram.
          if (diff_dram_x < 0 || diff_dram_y < 0) {
            target_dram_bank = dram;
            // If both target dram core and current dram core come before the worker, choose the one that's closer.
          } else if (diff_x + diff_y < diff_dram_x + diff_dram_y) {
            target_dram_bank = dram;
          }
        }
      }
    }
    if (soc_descriptor->perf_dram_bank_to_workers.find(target_dram_bank) == soc_descriptor->perf_dram_bank_to_workers.end()) {
      soc_descriptor->perf_dram_bank_to_workers.insert(std::pair<CoreCoord, std::vector<CoreCoord>>(target_dram_bank, {worker}));
    } else {
      soc_descriptor->perf_dram_bank_to_workers[target_dram_bank].push_back(worker);
    }
  }
}

std::unique_ptr<tt_SocDescriptor> load_soc_descriptor_from_yaml(std::string device_descriptor_file_path) {
  std::unique_ptr<tt_SocDescriptor> soc_descriptor = std::unique_ptr<tt_SocDescriptor>(new tt_SocDescriptor());

  std::ifstream fdesc(device_descriptor_file_path);
  if (fdesc.fail()) {
      throw std::runtime_error("Error: device descriptor file " + device_descriptor_file_path + " does not exist!");
  }
  fdesc.close();

  YAML::Node device_descriptor_yaml = YAML::LoadFile(device_descriptor_file_path);
  std::vector<std::size_t> trisc_sizes = {MEM_TRISC0_SIZE,
                                          MEM_TRISC1_SIZE,
                                          MEM_TRISC2_SIZE};  // TODO: Read trisc size from yaml

  auto grid_size_x = device_descriptor_yaml["grid"]["x_size"].as<int>();
  auto grid_size_y = device_descriptor_yaml["grid"]["y_size"].as<int>();

  load_core_descriptors_from_device_descriptor(device_descriptor_yaml, *soc_descriptor);
  soc_descriptor->grid_size = CoreCoord(grid_size_x, grid_size_y);
  soc_descriptor->device_descriptor_file_path = device_descriptor_file_path;
  soc_descriptor->trisc_sizes = trisc_sizes;

  std::string arch_name_value = device_descriptor_yaml["arch_name"].as<std::string>();
  arch_name_value = trim(arch_name_value);

  soc_descriptor->arch = tt::get_arch_from_string(arch_name_value);

  load_soc_features_from_device_descriptor(device_descriptor_yaml, soc_descriptor.get());

  map_workers_to_dram_banks(soc_descriptor.get());
  return soc_descriptor;
}

const std::string get_product_name(tt::ARCH arch, uint32_t num_harvested_noc_rows) {
  const static std::map<tt::ARCH, std::map<uint32_t, std::string> > product_name = {
      {tt::ARCH::GRAYSKULL, { {0, "E150"} } },
      {tt::ARCH::WORMHOLE_B0, { {0, "galaxy"}, {1, "nebula_x1"}, {2, "nebula_x2"} } }
  };

  return product_name.at(arch).at(num_harvested_noc_rows);
}

void load_dispatch_and_banking_config(tt_SocDescriptor &soc_descriptor, uint32_t num_harvested_noc_rows) {
  YAML::Node device_descriptor_yaml = YAML::LoadFile(soc_descriptor.device_descriptor_file_path);

  auto product_to_config  = device_descriptor_yaml["dispatch_and_banking"];
  auto product_name = get_product_name(soc_descriptor.arch, num_harvested_noc_rows);
  auto config = product_to_config[product_name];

  soc_descriptor.l1_bank_size = config["l1_bank_size"].as<int>();

  // TODO: Add validation for compute_with_storage, storage only, and dispatch core specification
  auto compute_with_storage_start = config["compute_with_storage_grid_range"]["start"];
  auto compute_with_storage_end = config["compute_with_storage_grid_range"]["end"];
  TT_ASSERT(compute_with_storage_start.IsSequence() and compute_with_storage_end.IsSequence());
  TT_ASSERT(compute_with_storage_end[0].as<size_t>() >= compute_with_storage_start[0].as<size_t>());
  TT_ASSERT(compute_with_storage_end[1].as<size_t>() >= compute_with_storage_start[1].as<size_t>());

  soc_descriptor.compute_with_storage_grid_size = CoreCoord({
    .x = (compute_with_storage_end[0].as<size_t>() - compute_with_storage_start[0].as<size_t>()) + 1,
    .y = (compute_with_storage_end[1].as<size_t>() - compute_with_storage_start[1].as<size_t>()) + 1,
  });

  // compute_with_storage_cores are a subset of worker cores
  // they have already been parsed as CoreType::WORKER and saved into `cores` map when parsing `functional_workers`
  for (auto x = 0; x < soc_descriptor.compute_with_storage_grid_size.x; x++) {
    for (auto y = 0; y < soc_descriptor.compute_with_storage_grid_size.y; y++) {
        const auto relative_coord = RelativeCoreCoord({.x = x, .y = y});
        soc_descriptor.compute_with_storage_cores.push_back(relative_coord);
    }
  }

  // storage_cores are a subset of worker cores
  // they have already been parsed as CoreType::WORKER and saved into `cores` map when parsing `functional_workers`
  for (const auto &core_node : config["storage_cores"]) {
    RelativeCoreCoord coord = {};
    if (core_node.IsSequence()) {
      // Logical coord
      coord = RelativeCoreCoord({.x = core_node[0].as<int>(), .y = core_node[1].as<int>()});
    } else {
      tt::log_fatal ("Only logical relative coords supported for storage_cores cores");
    }
    soc_descriptor.storage_cores.push_back(coord);
  }

  // dispatch_cores are a subset of worker cores
  // they have already been parsed as CoreType::WORKER and saved into `cores` map when parsing `functional_workers`
  for (const auto &core_node : config["dispatch_cores"]) {
    RelativeCoreCoord coord = {};
    if (core_node.IsSequence()) {
      // Logical coord
      coord = RelativeCoreCoord({.x = core_node[0].as<int>(), .y = core_node[1].as<int>()});
    } else {
      tt::log_fatal ("Only logical relative coords supported for dispatch_cores cores");
    }
    soc_descriptor.dispatch_cores.push_back(coord);
  }
}
