// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh8 device_target \"cpu\" axes [!distributed.physical_comm_axis<8, 1>]"' --distributed-search-strategies="beam-size=4 log-progress=true" %s -o /dev/null 2>&1 | FileCheck %s

// Checks that DistributedSearchStrategiesPass orders the fixed axis-decision
// sequence (LogicalAxisOrder) by a rough compute+communication importance
// estimate, rather than by findAllLogicalAxes' incidental module-walk order.
// This module ties one logical axis to a tiny elementwise add (16 flops, a
// few hundred bytes) and another to a large matmul (millions of flops, tens
// of megabytes of operand/result data): the add's axis is walked first
// (%arg0 precedes %arg2), so an unordered (or walk-ordered) search would
// decide it first, but the roofline+volume estimate should place the
// matmul's axis(es) ahead of it instead.
//
// log-progress's own remark ("parent at axis depth D ... extent remaining
// R") is reused as the observation point: R is the axis currently being
// decided's full remaining extent, which for a not-yet-touched axis equals
// its declared extent (2048 for the matmul's contracting dimension, 4 for
// the add's sharded dimension). That is distinct enough to identify which op
// an axis belongs to without depending on incidental total-axis-count or
// logical-axis-id details.
//
// CHECK: parent at axis depth 0 of {{[0-9]+}}, extent remaining 2048
// CHECK-NOT: axis depth 0 of {{[0-9]+}}, extent remaining 4,
// CHECK: extent remaining 4,

module {
  sdy.mesh @mesh = <["cheap"=4, "expensive"=2]>
  func.func @main(
      %arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"cheap"}, {}]>},
      %arg1: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
      %arg2: tensor<2x2048xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"expensive"}, {}]>},
      %arg3: tensor<2048x2048xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
      -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
          tensor<2x2048xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4x4xf32>
    %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0] :
      (tensor<2x2048xf32>, tensor<2048x2048xf32>) -> tensor<2x2048xf32>
    return %0, %1 : tensor<4x4xf32>, tensor<2x2048xf32>
  }
}
