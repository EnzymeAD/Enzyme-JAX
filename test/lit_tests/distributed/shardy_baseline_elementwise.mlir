// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | FileCheck %s
// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | enzymexlamlir-opt --distributed-search-lowering-pipeline="finalize-schedule=false" | FileCheck %s --check-prefix=LOWER

// A dimension Shardy shards over "a" (the physical axis of stride 2) gets that
// physical digit range as its major factor. Everything Shardy leaves unsharded
// is a device-local axis, so no logical axis remains for a search to decide.

module {
  sdy.mesh @mesh = <["a"=2, "b"=2]>
  func.func @main(%x: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
    %0 = stablehlo.negate %x {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// CHECK-NOT: distributed.LogicalMeshAxes
// CHECK-DAG: axis.factor %{{.*}} : !distributed.physical_comm_axis<2, 2><2, 1>
// CHECK-DAG: distributed.DeviceLocalAxis 4
// CHECK-DAG: distributed.DeviceLocalAxis 8
// CHECK-DAG: axis.product (%{{.*}} : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %{{.*}} : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
// CHECK: distributed.DistributedKernel

// The result lowers: the operand is split in two along dim 0 and nothing else
// needs communication.
// LOWER-NOT: distributed.DistributedKernel
// LOWER: stablehlo.negate %{{.*}} : tensor<4x8xf32>
