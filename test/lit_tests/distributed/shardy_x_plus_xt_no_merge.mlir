// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards --shardy-to-distributed-higher-level="dump-logical-axes=true" -split-input-file -o /dev/null %s 2>&1 | FileCheck %s

// 2D mesh sharding on both X axes.
module @x_plus_xt_2d {
  sdy.mesh @mesh = <["x"=2, "y"=2]>

  func.func @main(
      %x: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
      -> tensor<4x4xf32> {
    %xt = stablehlo.transpose %x, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %sum = stablehlo.add %x, %xt : tensor<4x4xf32>
    return %sum : tensor<4x4xf32>
  }
}

// CHECK: op: %{{.*}} = stablehlo.add %{{.*}} : tensor<4x4xf32>
// CHECK-NEXT:     partitioning axes: [a[[A2D:[0-9]+]]:4 ] [a[[B2D:[0-9]+]]:4 ]
// CHECK-NOT: partitioning axes: [a[[A2D]]:4 ] [a[[A2D]]:4 ]

// -----

// 1D mesh sharding on one X axis only.
module @x_plus_xt_1d {
  sdy.mesh @mesh = <["x"=2]>

  func.func @main(
      %x: tensor<6x6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
      -> tensor<6x6xf32> {
    %xt = stablehlo.transpose %x, dims = [1, 0] : (tensor<6x6xf32>) -> tensor<6x6xf32>
    %sum = stablehlo.add %x, %xt : tensor<6x6xf32>
    return %sum : tensor<6x6xf32>
  }
}

// CHECK: op: %{{.*}} = stablehlo.add %{{.*}} : tensor<6x6xf32>
// CHECK-NEXT:     partitioning axes: [a[[A1D:[0-9]+]]:6 ] [a[[B1D:[0-9]+]]:6 ]
// CHECK-NOT: partitioning axes: [a[[A1D]]:6 ] [a[[A1D]]:6 ]

// -----

// No sharding on either X axis.
module @x_plus_xt_none {
  sdy.mesh @mesh = <["x"=2]>

  func.func @main(
      %x: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
      -> tensor<8x8xf32> {
    %xt = stablehlo.transpose %x, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %sum = stablehlo.add %x, %xt : tensor<8x8xf32>
    return %sum : tensor<8x8xf32>
  }
}

// CHECK: op: %{{.*}} = stablehlo.add %{{.*}} : tensor<8x8xf32>
// CHECK-NEXT:     partitioning axes: [a[[ANONE:[0-9]+]]:8 ] [a[[BNONE:[0-9]+]]:8 ]
// CHECK-NOT: partitioning axes: [a[[ANONE]]:8 ] [a[[ANONE]]:8 ]
