// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --shardy-to-distributed-pipeline %s | FileCheck %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=8 fail-on-bad-candidate" %s 2>/dev/null | FileCheck %s --check-prefix=SEARCH

// A layer written once and called twice, whose body splits the hidden
// dimension of its activation (16 -> 4x4). Shardy shards the callee arguments
// differently from main's arguments, so propagation inserts reshards on main's
// arguments feeding the calls; each needs a collective before the call. Both
// call sites share the callee's axes.

module {
  sdy.mesh @mesh = <["tile"=2, "model"=2]>
  func.func private @layer(%h: tensor<8x16xf32>, %w: tensor<16x16xf32>, %cos: tensor<8x4x4xf32>) -> tensor<8x16xf32> {
    %0 = stablehlo.dot_general %h, %w, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
    %1 = stablehlo.reshape %0 : (tensor<8x16xf32>) -> tensor<8x4x4xf32>
    %2 = stablehlo.multiply %1, %cos : tensor<8x4x4xf32>
    %3 = stablehlo.reshape %2 : (tensor<8x4x4xf32>) -> tensor<8x16xf32>
    %4 = stablehlo.add %h, %3 : tensor<8x16xf32>
    return %4 : tensor<8x16xf32>
  }
  func.func @main(%x: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {}]>}, %w0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"model"}]>}, %w1: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"model"}]>}, %cos: tensor<8x4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}) -> tensor<8x16xf32> {
    %0 = func.call @layer(%x, %w0, %cos) : (tensor<8x16xf32>, tensor<16x16xf32>, tensor<8x4x4xf32>) -> tensor<8x16xf32>
    %1 = func.call @layer(%0, %w1, %cos) : (tensor<8x16xf32>, tensor<16x16xf32>, tensor<8x4x4xf32>) -> tensor<8x16xf32>
    return %1 : tensor<8x16xf32>
  }
}

// CHECK: sym_name = "main"
// CHECK: ^bb0(%[[X:.*]]: tensor<8x16xf32>, %{{.*}}: tensor<16x16xf32>, %{{.*}}: tensor<16x16xf32>, %[[COS:.*]]: tensor<8x4x4xf32>):
// CHECK-DAG: %[[COSL:.*]] = distributed.CastGlobalToLocal %[[COS]]
// CHECK-DAG: %[[XL:.*]] = distributed.CastGlobalToLocal %[[X]]
// CHECK: distributed.Collective %[[XL]] {{.*}} reduces ()
// CHECK: distributed.Collective %[[COSL]] {{.*}} reduces ()
// CHECK: distributed.DistributedCall @layer
// CHECK-NEXT: ->
// CHECK-NEXT: axes (%[[A0:[0-9]+]] : !axis.factor_group<8>, %[[A1:[0-9]+]] : !axis.factor_group<16>, %[[A2:[0-9]+]] : !axis.factor_group<4>, %[[A3:[0-9]+]] : !axis.factor_group<4>)
// CHECK: distributed.Collective %[[COSL]] {{.*}} reduces ()
// CHECK: distributed.DistributedCall @layer
// CHECK-NEXT: ->
// CHECK-NEXT: axes (%[[A0]] : !axis.factor_group<8>, %[[A1]] : !axis.factor_group<16>, %[[A2]] : !axis.factor_group<4>, %[[A3]] : !axis.factor_group<4>)
// CHECK: "distributed.DistributedFunction"(%[[A0]], %[[A1]], %[[A2]], %[[A3]])
// CHECK-SAME: sym_name = "layer"
// The split hidden dimension is one tensor dimension over two axes.
// CHECK: distributed.DistributedKernel {{.*}} <[<dim_partitioning_axes = {{\[}}[0], [1, 2]] : unreduced_axes = []>, <dim_partitioning_axes = {{\[}}[0], [1], [2]] : unreduced_axes = []>]>
// CHECK: stablehlo.reshape %{{.*}} : (tensor<8x16xf32>) -> tensor<8x4x4xf32>

// SEARCH: sym_name = "main"
// SEARCH: distributed.DistributedCall @layer
// SEARCH: distributed.DistributedCall @layer
// SEARCH: sym_name = "layer"
// SEARCH: stablehlo.reshape %{{.*}} : (tensor<8x16xf32>) -> tensor<8x4x4xf32>
