// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | FileCheck %s
// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | enzymexlamlir-opt --distributed-search-lowering-pipeline="finalize-schedule=false" | FileCheck %s --check-prefix=LOWER

// Merging [8, 4] into [32] with the major dimension sharded keeps the shard a
// major prefix of the merged dimension.

module {
  sdy.mesh @mesh = <["a"=2, "b"=2]>
  func.func @main(%x: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
    %0 = stablehlo.reshape %x {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : (tensor<8x4xf32>) -> tensor<32xf32>
    return %0 : tensor<32xf32>
  }
}

// CHECK-NOT: distributed.LogicalMeshAxes
// CHECK-NOT: distributed.Collective
// CHECK-DAG: axis.factor %{{.*}} : !distributed.physical_comm_axis<2, 2><2, 1>
// CHECK: stablehlo.reshape

// LOWER: stablehlo.reshape %{{.*}} : (tensor<4x4xf32>) -> tensor<16xf32>
