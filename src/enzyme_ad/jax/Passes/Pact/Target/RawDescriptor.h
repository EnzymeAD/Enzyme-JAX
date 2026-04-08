#ifndef PACT_TARGET_RAW_DESCRIPTOR_H
#define PACT_TARGET_RAW_DESCRIPTOR_H

#include "src/enzyme_ad/jax/Passes/Pact/PropertyScheme.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <map>
#include <string>

namespace mlir::enzyme::pact {

struct OccupancyEntry {
  int vgpr_threshold;
  int max_waves_per_simd;
};

struct NetworkPrimitive {
  scheme::MechanismKind type;
  int latency_cycles = 1;
  bool cross_row = false;
  int data_width_bits = 32;
};

struct WaveCompute {
  int width;
  llvm::SmallVector<int> width_alternatives;
  llvm::SmallVector<std::pair<int, scheme::ProgressCapability>, 2>
      progress_model_map;
  int simd_units_per_cu = 0;
};

struct WaveNetwork {
  llvm::SmallVector<NetworkPrimitive, 4> primitives;
  bool implicit_sync = false;
};

struct WaveMemory {
  int vgpr_per_simd = 0;
  int vgpr_alloc_granularity = 0;
  int sgpr_per_wave = 0;
  llvm::SmallVector<OccupancyEntry, 8> occupancy_table;

  int lds_bank_count = 0;
  int lds_bank_width_bytes = 0;
  int lds_latency_cycles = 0;
};

struct WaveScale {
  WaveCompute compute;
  WaveNetwork network;
  WaveMemory memory;
};

struct BlockScale {
  // TODO
};

struct ChipScale {
  // TODO
};

struct RawDescriptor {
  std::string target_id;
  WaveScale wave;
  BlockScale block;
  ChipScale chip;
};

} // namespace mlir::enzyme::pact
#endif // PACT_TARGET_RAW_DESCRIPTOR_H