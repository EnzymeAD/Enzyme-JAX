// RUN: enzymexlamlir-opt --shardy-to-distributed-pipeline %s | FileCheck %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=8 fail-on-bad-candidate" %s 2>/dev/null | FileCheck %s --check-prefix=SEARCH

// Search-mode lowering of reshards applied directly to function arguments:
// the query heads gain a factor ({"model"} -> {"model", "tile"}) and the key
// sequence is all-gathered ({"tile"} -> {}). Each reshard is a user-placed
// break, so its two sides are separate logical axes, and erasing it must leave
// a collective between the argument and its consumer.

module {
  sdy.mesh @mesh = <["tile"=2, "model"=2]>
  func.func @main(%q: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {}]>}, %k: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model", "tile"}, {}]>}) {
    %0 = sdy.reshard %q <@mesh, [{"model", "tile"}, {}]> : tensor<4x8xf32>
    %1 = sdy.reshard %k <@mesh, [{}, {}]> : tensor<16x8xf32>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"model", "tile"}, {}]>]>} : (tensor<4x8xf32>, tensor<16x8xf32>) -> tensor<4x16xf32>
    return %2 : tensor<4x16xf32>
  }
}

// The heads of %q and of the result are distinct axes (0 and 3), as are the
// key sequence of %k and the result's key axis (2 and 4). Only the shared
// head dimension 8 is one axis throughout.
// CHECK: "distributed.DistributedFunction"
// CHECK-SAME: argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = {{\[}}[0], [1]] : unreduced_axes = []>, <dim_partitioning_axes = {{\[}}[2], [1]] : unreduced_axes = []>]>
// CHECK-SAME: output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = {{\[}}[3], [4]] : unreduced_axes = []>]>
// CHECK: ^bb0(%[[Q:.*]]: tensor<4x8xf32>, %[[K:.*]]: tensor<16x8xf32>):
// CHECK: %[[KL:.*]] = distributed.CastGlobalToLocal %[[K]]
// CHECK: %[[QL:.*]] = distributed.CastGlobalToLocal %[[Q]]
// CHECK: distributed.Collective %[[QL]] : tensor<1x1xf32> on %{{.*}} : <32> to tensor<1x1xf32> on %{{.*}} : <32> reduces ()
// CHECK: distributed.Collective %[[KL]] : tensor<1x1xf32> on %{{.*}} : <128> to tensor<1x1xf32> on %{{.*}} : <128> reduces ()
// CHECK: distributed.DistributedKernel
// CHECK: stablehlo.dot_general
// The only other collective reduces the contraction.
// CHECK: distributed.Collective %{{.*}} reduces (%{{.*}} : !axis.factor_group<8>)
// CHECK-NOT: distributed.Collective

// SEARCH: "distributed.DistributedFunction"
// SEARCH: distributed.DistributedKernel
// SEARCH: stablehlo.dot_general
// SEARCH: distributed.DistributedYield
