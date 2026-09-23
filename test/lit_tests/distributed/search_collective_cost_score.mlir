// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=100 dump-candidates dump-best" %s -o /dev/null 2>&1 | FileCheck %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=100 dump-candidates" %s -o /dev/null 2> %t.first
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=100 dump-candidates" %s -o /dev/null 2> %t.second
// RUN: diff %t.first %t.second

// The search scores a candidate by the negated makespan a timeline simulation
// predicts for it (ScheduleGraph.h/ScheduleSimulator.h), so the scores are
// deterministic and can be checked by hand. This matmul contracts a dimension
// sharded 4 ways, so a candidate that keeps the contraction distributed needs
// an all-reduce of the 64x64 f32 result, and one that replicates everything
// needs none. Every candidate here has its kernel and its collective (if any)
// on one straight-line dependency chain (a cast feeds the kernel, the kernel
// feeds the all-reduce), so there is nothing to overlap and the makespan
// equals the plain sum of the collective and compute durations. Network
// costs use the default mesh parameters
// (bandwidth 1, round latency 0.01, launch 0.1), so a step that moves V bytes
// in one round costs V + 0.11:
//
//   - a 2-way all-reduce of the 8192-byte 32x64 tile: 8192 + 0.11 (recursive
//     doubling, one round)
//   - a 2-way all-reduce of the 16384-byte 64x64 tile: 16384 + 0.11
//   - the 4-way all-reduce of the 16384-byte 64x64 tile, planned over two
//     2-way atoms as reduce-scatter, all-reduce, all-gather: three steps of
//     8192 + 0.11 each, 24576.33
//   - fully local candidates have no collectives.
//
// The mesh declares device_flops = 1e6 and device_mem_bandwidth = 1e9, so the
// matmul is compute bound (its bytes / 1e9 is far below its flops / 1e6) and
// costs 2 * M * N * K / 1e6 at its local shape:
//
//   - unsharded 64x128 . 128x64:     2 * 64 * 64 * 128 / 1e6 = 1.048576
//   - two-way sharded, e.g. K = 64:  2 * 64 * 64 * 64 / 1e6  = 0.524288
//   - four-way sharded, e.g. N = 16: 2 * 64 * 16 * 128 / 1e6 = 0.262144
//
// Compute is small next to the network cost here, so a candidate with
// collectives is dominated by them, and among the collective-free candidates
// the cheapest compute wins.
//
// A candidate with a collective always reports two more nodes than its step
// count would suggest: distributed-lower-collectives (unconditional for
// every scored candidate, not only the eventual winner) flattens the
// collective's operand to 1-D before its chain and reshapes the chain's
// result back afterward, and both reshapes are zero-cost Movement ops
// (ComputeCost.h) that never change a makespan already dominated by network
// time.

// Candidates with no collective.
// CHECK-DAG: // Search candidate (ok, score=-1.048576e+00):
// CHECK-DAG: // schedule cost: makespan=1.048576e+00 over 1 nodes (communicating=0.000000e+00 computing=1.048576e+00 utilization=1.000000e+00 idle=0.000000e+00)
// CHECK-DAG: // Search candidate (ok, score=-5.242880e-01):
// CHECK-DAG: // schedule cost: makespan=5.242880e-01 over 1 nodes (communicating=0.000000e+00 computing=5.242880e-01 utilization=1.000000e+00 idle=0.000000e+00)
// CHECK-DAG: // Search candidate (ok, score=-2.621440e-01):
// CHECK-DAG: // schedule cost: makespan=2.621440e-01 over 1 nodes (communicating=0.000000e+00 computing=2.621440e-01 utilization=1.000000e+00 idle=0.000000e+00)
// One 2-way all-reduce of the 32x64 tile: 8192.11 + 0.262144, on the same
// dependency chain as the kernel it follows, so nothing overlaps it.
// CHECK-DAG: // Search candidate (ok, score=-8.192372e+03):
// CHECK-DAG: // schedule cost: makespan=8.192372e+03 over 4 nodes (communicating=8.192110e+03 computing=2.621440e-01 utilization=3.199855e-05 idle=8.192110e+03)
// One 2-way all-reduce of the 64x64 tile: 16384.11 + 0.524288.
// CHECK-DAG: // Search candidate (ok, score=-1.638463e+04):
// CHECK-DAG: // schedule cost: makespan=1.638463e+04 over 4 nodes (communicating=1.638411e+04 computing=5.242880e-01 utilization=3.199876e-05 idle=1.638411e+04)
// The 4-way all-reduce: 24576.33 + 0.262144.
// CHECK-DAG: // Search candidate (ok, score=-2.457659e+04):
// CHECK-DAG: // schedule cost: makespan=2.457659e+04 over 6 nodes (communicating=2.457633e+04 computing=2.621440e-01 utilization=1.066641e-05 idle=2.457633e+04)
// The best candidate needs no collective and shards the compute four ways.
// CHECK-DAG: // Best search candidate (ok, score=-2.621440e-01):

module {
  sdy.mesh @mesh = <["x"=4]>
  func.func @main(
      %arg0: tensor<64x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
      %arg1: tensor<128x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
      -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] :
      (tensor<64x128xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
