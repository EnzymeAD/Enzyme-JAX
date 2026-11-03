// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | FileCheck %s
// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | enzymexlamlir-opt --distributed-search-lowering-pipeline="finalize-schedule=false" | FileCheck %s --check-prefix=LOWER

// Grouped-query attention's KV repeat: 2 KV heads sharded over "model" are
// broadcast to a [2, 2] group (the new group dim sharded over "tile") and
// reshaped to 4 heads sharded over ["model", "tile"], matching the 4 query
// heads. Both dot operands' head dims then carry the same [model, tile]
// factorization with the model symbol in common, and must unify into one
// logical axis: each device already holds matching Q and K heads, so the
// repeat stays in the attention kernel and no collective is needed.

module {
  sdy.mesh @mesh = <["tile"=2, "model"=2]>
  func.func @main(%q: tensor<4x1x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model", "tile"}, {}, {}]>}, %k: tensor<2x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {}, {}]>}) -> (tensor<4x1x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model", "tile"}, {}, {}]>}) {
    %0 = stablehlo.broadcast_in_dim %k, dims = [0, 2, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"model"}, {"tile"}, {}, {}]>]>} : (tensor<2x16x8xf32>) -> tensor<2x2x16x8xf32>
    %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"model", "tile"}, {}, {}]>]>} : (tensor<2x2x16x8xf32>) -> tensor<4x16x8xf32>
    %2 = stablehlo.dot_general %q, %1, batching_dims = [0] x [0], contracting_dims = [2] x [2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"model", "tile"}, {}, {}]>]>} : (tensor<4x1x8xf32>, tensor<4x16x8xf32>) -> tensor<4x1x16xf32>
    return %2 : tensor<4x1x16xf32>
  }
}

// CHECK-NOT: distributed.Collective
// CHECK: distributed.DistributedKernel
// CHECK: axes (%{{.*}} : !axis.factor_group<2>, %{{.*}} : !axis.factor_group<16>, %{{.*}} : !axis.factor_group<8>, %{{.*}} : !axis.factor_group<2>)
// The broadcast's group dim, the repeated heads, and both dot operands' heads
// all use slot 3 for the "tile" digits.
// CHECK: stablehlo.broadcast_in_dim
// CHECK-SAME: distributed.output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = {{\[}}[0], [3], [1], [2]]
// CHECK: stablehlo.reshape
// CHECK-SAME: distributed.output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = {{\[}}[0, 3], [1], [2]]
// CHECK: stablehlo.dot_general
// CHECK-SAME: distributed.argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = {{\[}}[0, 3], [], [2]] : unreduced_axes = []>, <dim_partitioning_axes = {{\[}}[0, 3], [1], [2]]
// CHECK: distributed.DistributedYield
// CHECK-NOT: distributed.DistributedKernel
// CHECK-NOT: distributed.Collective

// Each device holds one KV head and repeats it locally into its one query head.
// LOWER-NOT: distributed.Collective
// LOWER: stablehlo.broadcast_in_dim %{{.*}}, dims = [0, 2, 3] {{.*}}: (tensor<1x16x8xf32>) -> tensor<1x1x16x8xf32>
// LOWER: stablehlo.reshape %{{.*}} : (tensor<1x1x16x8xf32>) -> tensor<1x16x8xf32>
// LOWER: stablehlo.dot_general %{{.*}}, %{{.*}}, batching_dims = [0] x [0], contracting_dims = [2] x [2] {{.*}}: (tensor<1x1x8xf32>, tensor<1x16x8xf32>) -> tensor<1x1x16xf32>
// LOWER-NOT: distributed.Collective
