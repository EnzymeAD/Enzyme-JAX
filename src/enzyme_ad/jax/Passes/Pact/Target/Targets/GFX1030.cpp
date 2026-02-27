#include "../RawDescriptor.h"
#include "../TargetRegistry.h"

namespace mlir::enzyme::pact {

static bool registered = [] {
  using namespace pact;

  RawDescriptor desc;
  desc.target_id = "gfx1030";

  // WaveScale.compute
  desc.wave.compute.width = 32;
  desc.wave.compute.width_alternatives = {32, 64};
  // TODO
  // desc.wave.compute.simd_units_per_cu = 2;
  // desc.wave.compute.progress_model_by_wave_size = {
  //     {32, ProgressModel::LockstepWithinWave},
  //     {64, ProgressModel::LockstepWithinWave},
  // };

  // WaveScale.memory
  desc.wave.memory.vgpr_per_simd = 1024;
  desc.wave.memory.vgpr_alloc_granularity = 16;
  // TODO
  // sgpr_per_wave = 128
  // occupancy_table

  // TODO

  TargetRegistry::registerTarget("gfx1030", std::move(desc));
  return true;
}();

} // namespace mlir::enzyme::pact