// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | FileCheck %s
// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | enzymexlamlir-opt --distributed-search-lowering-pipeline="finalize-schedule=false" | FileCheck %s --check-prefix=LOWER

// An explicit reshard between two Shardy shardings becomes a collective whose
// two sides are the physical digit ranges of "a" and "b".

module {
  sdy.mesh @mesh = <["a"=2, "b"=2]>
  func.func @main(%x: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>}) {
    %0 = stablehlo.negate %x {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x8xf32>
    %1 = sdy.reshard %0 <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
    %2 = stablehlo.negate %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : tensor<8x8xf32>
    return %2 : tensor<8x8xf32>
  }
}

// CHECK-NOT: distributed.LogicalMeshAxes
// CHECK-DAG: axis.factor %{{.*}} : !distributed.physical_comm_axis<2, 2><2, 1>
// CHECK-DAG: axis.factor %{{.*}} : !distributed.physical_comm_axis<2, 1><2, 1>
// CHECK: distributed.Collective

// LOWER-NOT: distributed.Collective
// LOWER: stablehlo.negate %{{.*}} : tensor<4x8xf32>
// LOWER: stablehlo.negate %{{.*}} : tensor<8x4xf32>
