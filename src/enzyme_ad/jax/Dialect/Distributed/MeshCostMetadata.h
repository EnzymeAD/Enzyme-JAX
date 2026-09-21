#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_MESH_COST_METADATA_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_MESH_COST_METADATA_H

#include "CollectiveCost.h"
#include "Dialect.h"

namespace mlir::enzyme::distributed {

// Cost parameters read from the optional performance attributes of a
// PhysicalMeshOp. Every attribute that is absent falls back to an abstract
// default.
//
// Units are abstract. All numbers of one mesh live in one consistent but
// arbitrary system (bytes per time unit, FLOP per time unit, time units), and
// only the relative strength of compute versus communication affects a search.

// Per-axis bandwidth and round latency and the launch latency of `mesh`, with
// axis extents taken from its axes. Defaults are the MeshCostParams constants.
MeshCostParams meshCostParamsFromPhysicalMesh(PhysicalMeshOp mesh);

// Peak throughput of one device.
//
// Defaults are chosen relative to the default network bandwidth of 1 byte per
// time unit: a device with a memory bandwidth 20 times, and a peak FLOP rate
// 2000 times, that of one network port. That is the ratio of a device with
// roughly 100 TFLOP/s and 1 TB/s of memory bandwidth to a 50 GB/s link, and
// the balance point (flops / memBandwidth) of 100 FLOP per byte is typical of
// accelerators.
struct DeviceCostParams {
  // Peak FLOP per time unit.
  double flops;
  // Peak local memory bytes per time unit.
  double memBandwidth;

  static constexpr double kDefaultFlops = 2000.0;
  static constexpr double kDefaultMemBandwidth = 20.0;

  DeviceCostParams(double flops = kDefaultFlops,
                   double memBandwidth = kDefaultMemBandwidth)
      : flops(flops), memBandwidth(memBandwidth) {}
};

DeviceCostParams deviceCostParamsFromPhysicalMesh(PhysicalMeshOp mesh);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_MESH_COST_METADATA_H
