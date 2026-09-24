// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | FileCheck %s
// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | enzymexlamlir-opt --distributed-search-lowering-pipeline="finalize-schedule=false" | FileCheck %s --check-prefix=LOWER

// Each sdy.mesh is projected onto the one physical mesh. @m1's "a" and the
// major half of @m2's "c" are the same physical digit range, so an op reading
// values sharded over each of them needs no communication.

module {
  sdy.mesh @m1 = <["a"=2, "b"=2]>
  sdy.mesh @m2 = <["c"=4]>
  func.func @main(%x: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@m1, [{"a"}, {}]>}, %y: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@m2, [{"c":(1)2}, {}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@m1, [{"a"}, {}]>}) {
    %0 = stablehlo.add %x, %y {sdy.sharding = #sdy.sharding_per_value<[<@m1, [{"a"}, {}]>]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// CHECK-NOT: distributed.LogicalMeshAxes
// CHECK-NOT: distributed.Collective
// CHECK-DAG: axis.factor %{{.*}} : !distributed.physical_comm_axis<2, 2><2, 1>
// CHECK: distributed.DistributedKernel

// LOWER: stablehlo.add %{{.*}}, %{{.*}} : tensor<4x8xf32>
