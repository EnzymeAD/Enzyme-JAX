// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | FileCheck %s
// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | enzymexlamlir-opt --distributed-search-lowering-pipeline="finalize-schedule=false" | FileCheck %s --check-prefix=LOWER

// The contraction dimension is sharded over "a", so the dot produces partial
// sums that are all-reduced over that physical digit range. "b" shards the
// output columns.

module {
  sdy.mesh @mesh = <["a"=2, "b"=2]>
  func.func @main(%x: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>}, %w: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>}) {
    %0 = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// CHECK-NOT: distributed.LogicalMeshAxes
// CHECK-DAG: axis.factor %{{.*}} : !distributed.physical_comm_axis<2, 2><2, 1>
// CHECK-DAG: axis.factor %{{.*}} : !distributed.physical_comm_axis<2, 1><2, 1>
// The reduced axis is exactly the "a" digit range.
// CHECK: distributed.Collective
// CHECK-SAME: reduces (%{{.*}} : !axis.factor_group<2>)

// Each device holds an 8x8 slice of the activations and an 8x4 slice of the
// weights.
// LOWER: stablehlo.dot_general %{{.*}}, %{{.*}}, contracting_dims = [1] x [0] : (tensor<8x8xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>
// LOWER: stablehlo.all_reduce
