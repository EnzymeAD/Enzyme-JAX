// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | FileCheck %s
// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | enzymexlamlir-opt --distributed-search-lowering-pipeline="finalize-schedule=false" | FileCheck %s --check-prefix=LOWER

// Shardy does not insert a reshard when an op shards a dimension further than
// its operand is: each device slices its operand locally. The baseline keeps
// the operand's and the op's axes apart, so that slice is materialized as a
// collective from the replicated layout to the sharded one.

module {
  sdy.mesh @mesh = <["data"=2, "tp"=4]>
  func.func @main(%x: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tp"}]>}) {
    %0 = stablehlo.slice %x [0:8, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tp"}]>]>} : (tensor<8x16xf32>) -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
}

// CHECK-NOT: distributed.LogicalMeshAxes
// CHECK: distributed.Collective
// CHECK: distributed.DistributedKernel

// LOWER-NOT: distributed.Collective
// LOWER: stablehlo.slice %{{.*}} [0:4, 0:4] : (tensor<4x4xf32>) -> tensor<4x4xf32>
