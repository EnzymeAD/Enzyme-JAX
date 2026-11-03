#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_MESH_COST_METADATA_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_MESH_COST_METADATA_H

#include "CollectiveCost.h"
#include "Dialect.h"

namespace mlir::enzyme::distributed {

// Cost parameters read from the optional performance attributes of a
// PhysicalMeshOp. Every attribute that is absent falls back to an SI-derived
// default.
//
// The IR-level attributes (Ops.td) stay plain SI doubles (bytes, seconds,
// FLOP/s, bytes/s) for readability in a mesh string; meshCostParams/
// deviceCostParamsFromPhysicalMesh convert them once, here, into the integer
// nanosecond/bytes-per-nanosecond representation CollectiveCost.h's footprint
// math and the schedule simulator use throughout.

// Per-axis bandwidth and round latency and the launch latency of `mesh`, with
// axis extents taken from its axes. Defaults are the MeshCostParams constants.
MeshCostParams meshCostParamsFromPhysicalMesh(PhysicalMeshOp mesh);

// Peak throughput of one device.
//
// Defaults describe an H100 class accelerator at FP32/TF32 precision: about
// 100 TFLOP/s and 3 TB/s of HBM bandwidth (a balance point of about 33 FLOP
// per byte). Against the default 50 GB/s network port this is 2000 FLOP per
// network byte.
struct DeviceCostParams {
  // Peak FLOP per nanosecond.
  double flops;
  // Peak local memory bytes per nanosecond.
  double memBandwidth;

  static constexpr double kDefaultFlops = 1.0e5;
  static constexpr double kDefaultMemBandwidth = 3.0e3;

  DeviceCostParams(double flops = kDefaultFlops,
                   double memBandwidth = kDefaultMemBandwidth)
      : flops(flops), memBandwidth(memBandwidth) {}
};

DeviceCostParams deviceCostParamsFromPhysicalMesh(PhysicalMeshOp mesh);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_MESH_COST_METADATA_H
