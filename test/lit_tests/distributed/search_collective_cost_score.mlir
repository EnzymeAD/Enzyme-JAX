// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=100 dump-candidates dump-best" %s -o /dev/null 2>&1 | FileCheck %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=100 dump-candidates" %s -o /dev/null 2> %t.first
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=100 dump-candidates" %s -o /dev/null 2> %t.second
// RUN: diff %t.first %t.second

// The search scores a candidate by the negated sum of its collectives'
// isolated durations and its kernels' roofline compute times, so the scores
// are deterministic and can be checked by hand. This matmul contracts a
// dimension sharded 4 ways, so a candidate that keeps the contraction
// distributed needs an all-reduce of the 64x64 f32 result, and one that
// replicates everything needs none. Network costs use the default mesh
// parameters (bandwidth 1, round latency 0.01, launch 0.1), so a step that
// moves V bytes in one round costs V + 0.11:
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

// Candidates with no collective.
// CHECK-DAG: // Search candidate (ok, score=-1.048576e+00):
// CHECK-DAG: // collective cost: total=0.000000e+00 over 0 collectives:
// CHECK-DAG: // compute cost: total=1.048576e+00 over 1 kernels (unknown ops: 0, collectives: 0, skipped: 0)
// CHECK-DAG: // Search candidate (ok, score=-5.242880e-01):
// CHECK-DAG: // compute cost: total=5.242880e-01 over 1 kernels
// CHECK-DAG: // Search candidate (ok, score=-2.621440e-01):
// CHECK-DAG: // compute cost: total=2.621440e-01 over 1 kernels
// One 2-way all-reduce of the 32x64 tile: 8192.11 + 0.262144.
// CHECK-DAG: // Search candidate (ok, score=-8.192372e+03):
// CHECK-DAG: // collective cost: total=8.192110e+03 over 1 collectives: 8.192110e+03x1
// One 2-way all-reduce of the 64x64 tile: 16384.11 + 0.524288.
// CHECK-DAG: // Search candidate (ok, score=-1.638463e+04):
// CHECK-DAG: // collective cost: total=1.638411e+04 over 1 collectives: 1.638411e+04x1
// The 4-way all-reduce: 24576.33 + 0.262144.
// CHECK-DAG: // Search candidate (ok, score=-2.457659e+04):
// CHECK-DAG: // collective cost: total=2.457633e+04 over 1 collectives: 2.457633e+04x1
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
