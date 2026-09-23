// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=200 dump-candidates" %s -o /dev/null 2>&1 | FileCheck %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=200 dump-candidates" %s -o /dev/null 2> %t.first
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=200 dump-candidates" %s -o /dev/null 2> %t.second
// RUN: diff %t.first %t.second

// This module is the independent fixture from search_schedule_overlap_score.mlir
// (see there for how its makespans are derived), reused here only for its
// shape. Two logical axes ("x" and "y", both extent 2) mean a search node that
// has just decided "x" is not yet finalized, so it is scored and dumped
// (dump-candidates) and then expanded again to decide "y": a two-generation
// deep branch.
//
// Every StrategySearchNode gets a process-global, atomically minted uid (id 1
// is the initial root, which is never scored or dumped). The first generation
// below binds "x" and reports parent_uid=1 (the root). The second generation
// binds "y" on top of one particular first-generation candidate and must
// report that candidate's uid as its parent, not the root's or a sibling's.
// That is the lineage check.
//
// First generation: three candidates for how "x" gets bound (chunk from one
// physical factor, chunk from the other, or fully serialized within-device).
// CHECK: // uid=2 parent_uid=1
// CHECK: // uid=3 parent_uid=1

// Second generation: uid=2 and uid=3 each get expanded to decide "y", and
// their children must point back to the right parent (uid=5 came from uid=2,
// uid=8 from uid=3).
// CHECK: // uid=5 parent_uid=2
// CHECK: // uid=8 parent_uid=3

module {
  sdy.mesh @mesh = <["x"=2, "y"=2]>
  func.func @main(
      %arg0: tensor<1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
      %arg1: tensor<2x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      %arg2: tensor<1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>},
      %arg3: tensor<2x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>})
      -> (tensor<1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
          tensor<1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] :
      (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<1x1xf32>
    %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0] :
      (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<1x1xf32>
    return %0, %1 : tensor<1x1xf32>, tensor<1x1xf32>
  }
}
