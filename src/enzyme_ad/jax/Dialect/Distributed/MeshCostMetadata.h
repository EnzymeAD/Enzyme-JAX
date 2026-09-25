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
// FLOP/s, bytes/s, Hz) for readability in a mesh string; meshCostParams/
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
// per byte) at a 1.5 GHz clock. Against the default 50 GB/s network port this
// is 2000 FLOP per network byte.
struct DeviceCostParams {
  // Peak FLOP per nanosecond.
  double flops;
  // Peak local memory bytes per nanosecond.
  double memBandwidth;
  // Cycles per nanosecond, i.e. the device's clock rate. Nothing in the
  // roofline model itself is denominated in cycles; this parameter exists
  // solely to convert an ACT-reported act.cost (a cycle count) into
  // nanoseconds (see actCyclesToNs and ScheduleGraph.cpp's act.cost
  // handling).
  double clockFrequency;
  // Dimensionless factor applied to every act.cost before the clock
  // conversion. Calibrates ACT's simulator against this model, e.g. when ACT
  // simulates a part with a different clock, PE count or memory system than
  // the modeled device, or is systematically optimistic next to the
  // roofline estimate of the same kernel.
  double actCostScale;
  // Nanoseconds of fixed host-side cost charged for every real ACT kernel
  // launch (see appendActCalleeNode), on top of the launch's own act.cost.
  // Zero by default: unset, a candidate that dispatches many small ACT
  // launches instead of one large one pays no more host overhead than the
  // large one, which understates its real cost relative to a real
  // accelerator.
  double actKernelLaunchOverheadNs;

  static constexpr double kDefaultFlops = 1.0e5;
  static constexpr double kDefaultMemBandwidth = 3.0e3;
  static constexpr double kDefaultClockFrequency = 1.5;
  static constexpr double kDefaultActCostScale = 1.0;
  static constexpr double kDefaultActKernelLaunchOverheadNs = 0.0;

  DeviceCostParams(double flops = kDefaultFlops,
                   double memBandwidth = kDefaultMemBandwidth,
                   double clockFrequency = kDefaultClockFrequency,
                   double actCostScale = kDefaultActCostScale,
                   double actKernelLaunchOverheadNs =
                       kDefaultActKernelLaunchOverheadNs)
      : flops(flops), memBandwidth(memBandwidth),
        clockFrequency(clockFrequency), actCostScale(actCostScale),
        actKernelLaunchOverheadNs(actKernelLaunchOverheadNs) {}

  // Nanoseconds of an ACT launch that ACT costed at `cycles`, rounded to the
  // nearest nanosecond.
  int64_t actCyclesToNs(double cycles) const {
    return std::llround(cycles * actCostScale / clockFrequency);
  }
};

DeviceCostParams deviceCostParamsFromPhysicalMesh(PhysicalMeshOp mesh);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_MESH_COST_METADATA_H
