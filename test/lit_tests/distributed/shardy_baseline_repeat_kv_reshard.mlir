// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | FileCheck %s --implicit-check-not="reduces (%"
// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | enzymexlamlir-opt --distributed-search-lowering-pipeline="finalize-schedule=false" | FileCheck %s --check-prefix=LOWER --implicit-check-not=all_reduce

// Grouped-query attention scores from a fully sharded KV cache: keys and mask
// are stored with the sequence sharded over "tile" and are all-gathered, and
// the query heads are sliced from "model" to ["model", "tile"], so the KV
// repeat and the scores can run per head.
//
// - The function signature binds the key and mask sequence digits with
//   separate but identical physical factors, which CSE may merge. The casts
//   binding each argument must still unify with the signature.
// - After the gathers, the sequence dimension is unsharded, so the max over
//   it is device-local and needs no collective.

module {
  sdy.mesh @mesh = <["tile"=2, "model"=2]>
  func.func @main(%q: tensor<4x1x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {}, {}]>}, %k: tensor<2x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"tile"}, {}]>}, %mask: tensor<1x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"tile"}]>}) -> (tensor<4x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model", "tile"}, {}]>}) {
    %0 = sdy.reshard %q <@mesh, [{"model", "tile"}, {}, {}]> : tensor<4x1x8xf32>
    %1 = sdy.reshard %k <@mesh, [{"model"}, {}, {}]> : tensor<2x16x8xf32>
    %2 = sdy.reshard %mask <@mesh, [{}, {}]> : tensor<1x16xf32>
    %3 = stablehlo.broadcast_in_dim %1, dims = [0, 2, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"model"}, {"tile"}, {}, {}]>]>} : (tensor<2x16x8xf32>) -> tensor<2x2x16x8xf32>
    %4 = stablehlo.reshape %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"model", "tile"}, {}, {}]>]>} : (tensor<2x2x16x8xf32>) -> tensor<4x16x8xf32>
    %5 = stablehlo.dot_general %0, %4, batching_dims = [0] x [0], contracting_dims = [2] x [2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"model", "tile"}, {}, {}]>]>} : (tensor<4x1x8xf32>, tensor<4x16x8xf32>) -> tensor<4x1x16xf32>
    %6 = stablehlo.broadcast_in_dim %2, dims = [1, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"model", "tile"}, {}, {}]>]>} : (tensor<1x16xf32>) -> tensor<4x1x16xf32>
    %7 = stablehlo.add %5, %6 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"model", "tile"}, {}, {}]>]>} : tensor<4x1x16xf32>
    %8 = sdy.constant dense<0xFF800000> : tensor<f32>
    %9 = stablehlo.reduce(%7 init: %8) applies stablehlo.maximum across dimensions = [2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"model", "tile"}, {}]>]>} : (tensor<4x1x16xf32>, tensor<f32>) -> tensor<4x1xf32>
    return %9 : tensor<4x1xf32>
  }
}

// The three reshards are the only collectives, and none reduces.
// CHECK: distributed.Collective %{{.*}} : tensor<1x1x1xf32> on %{{.*}} : <256> to tensor<1x1x1xf32> on %{{.*}} : <256> reduces ()
// CHECK: distributed.Collective %{{.*}} : tensor<1x1x1xf32> on %{{.*}} : <32> to tensor<1x1x1xf32> on %{{.*}} : <32> reduces ()
// CHECK: distributed.Collective %{{.*}} : tensor<1x1xf32> on %{{.*}} : <16> to tensor<1x1xf32> on %{{.*}} : <16> reduces ()
// The KV repeat, the scores, and the max all share one kernel.
// CHECK: distributed.DistributedKernel ()
// CHECK: sdy.constant
// CHECK: distributed.DistributedKernel (
// CHECK: stablehlo.broadcast_in_dim %{{.*}}, dims = [0, 2, 3]
// CHECK: stablehlo.reshape
// CHECK: stablehlo.dot_general
// CHECK: stablehlo.add
// CHECK: stablehlo.reduce
// CHECK: distributed.DistributedYield
// CHECK-NOT: distributed.DistributedKernel
// CHECK-NOT: distributed.Collective

// LOWER: stablehlo.all_gather
// LOWER: stablehlo.dynamic_slice
// LOWER: stablehlo.all_gather
// LOWER: stablehlo.broadcast_in_dim %{{.*}}, dims = [0, 2, 3] : (tensor<1x16x8xf32>) -> tensor<1x1x16x8xf32>
// LOWER: stablehlo.dot_general %{{.*}}, %{{.*}}, batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<1x1x8xf32>, tensor<1x16x8xf32>) -> tensor<1x1x16xf32>
// LOWER: stablehlo.reduce(%{{.*}} init: %{{.*}}) applies stablehlo.maximum across dimensions = [2] : (tensor<1x1x16xf32>, tensor<f32>) -> tensor<1x1xf32>
