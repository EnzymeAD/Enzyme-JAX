// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | FileCheck %s
// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | enzymexlamlir-opt --distributed-search-lowering-pipeline="finalize-schedule=false" | FileCheck %s --check-prefix=LOWER

// Two dimensions shard over complementary sub-axes of the same Shardy axis "x"
// (size 4). Sub-axis (1)2 is the major digit of the physical axis and (2)2 the
// minor digit.

module {
  sdy.mesh @mesh = <["x"=4, "y"=2]>
  func.func @main(%a: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"x":(2)2}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"x":(2)2}]>}) {
    %0 = stablehlo.negate %a {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"x":(2)2}]>]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// CHECK-NOT: distributed.LogicalMeshAxes
// CHECK-DAG: axis.factor %{{.*}} : !distributed.physical_comm_axis<4, 2><2, 2>
// CHECK-DAG: axis.factor %{{.*}} : !distributed.physical_comm_axis<4, 2><2, 1>

// LOWER: stablehlo.negate %{{.*}} : tensor<4x4xf32>
