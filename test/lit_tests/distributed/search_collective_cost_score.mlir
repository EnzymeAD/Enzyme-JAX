// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>]"' --distributed-search-strategies="beam-size=100 dump-candidates dump-best" %s -o /dev/null 2>&1 | FileCheck %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>]"' --distributed-search-strategies="beam-size=100 dump-candidates" %s -o /dev/null 2> %t.first
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>]"' --distributed-search-strategies="beam-size=100 dump-candidates" %s -o /dev/null 2> %t.second
// RUN: diff %t.first %t.second

// The search scores a candidate by the negated sum of its collectives'
// isolated durations, so the scores are deterministic and can be checked by
// hand. This matmul contracts a dimension sharded 4 ways, so a candidate that
// keeps the contraction distributed needs an all-reduce of the 64x64 f32
// result, and one that replicates everything needs none. Costs use uniform
// mesh parameters (bandwidth 1, round latency 0.01, launch 0.1), so a step
// that moves V bytes in one round costs V + 0.11:
//
//   - a 2-way all-reduce of the 8192-byte 32x64 tile: 8192 + 0.11 (recursive
//     doubling, one round)
//   - a 2-way all-reduce of the 16384-byte 64x64 tile: 16384 + 0.11
//   - the 4-way all-reduce of the 16384-byte 64x64 tile, planned over two
//     2-way atoms as reduce-scatter, all-reduce, all-gather: three steps of
//     8192 + 0.11 each, 24576.33
//   - fully local candidates have no collectives and cost 0.

// CHECK-DAG: // Search candidate (ok, score=0.000000e+00):
// CHECK-DAG: // collective cost: total=0.000000e+00 over 0 collectives:
// CHECK-DAG: // Search candidate (ok, score=-8.192110e+03):
// CHECK-DAG: // collective cost: total=8.192110e+03 over 1 collectives: 8.192110e+03x1
// CHECK-DAG: // Search candidate (ok, score=-1.638411e+04):
// CHECK-DAG: // collective cost: total=1.638411e+04 over 1 collectives: 1.638411e+04x1
// CHECK-DAG: // Search candidate (ok, score=-2.457633e+04):
// CHECK-DAG: // collective cost: total=2.457633e+04 over 1 collectives: 2.457633e+04x1
// The cheapest candidate wins.
// CHECK-DAG: // Best search candidate (ok, score=0.000000e+00):

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
