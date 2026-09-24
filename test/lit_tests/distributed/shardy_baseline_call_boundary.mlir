// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | FileCheck %s
// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | enzymexlamlir-opt --distributed-search-lowering-pipeline="finalize-schedule=false" | FileCheck %s --check-prefix=LOWER

// The callee expects its argument sharded over "tp" while the caller passes it
// replicated over "tp". The two sides keep separate axes, so the difference is
// a collective at the call.

module {
  sdy.mesh @mesh = <["data"=2, "tp"=4]>
  func.func private @f(%a: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tp"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tp"}]>}) {
    %0 = stablehlo.negate %a {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tp"}]>]>} : tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
  func.func @main(%x: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tp"}]>}) {
    %0 = call @f(%x) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tp"}]>]>} : (tensor<8x16xf32>) -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
}

// CHECK-NOT: distributed.LogicalMeshAxes
// CHECK: distributed.DistributedFunction
// CHECK: distributed.DistributedFunction
// CHECK: distributed.Collective
// CHECK: distributed.DistributedCall @f

// LOWER-NOT: distributed.Collective
// LOWER: stablehlo.negate %{{.*}} : tensor<4x4xf32>
