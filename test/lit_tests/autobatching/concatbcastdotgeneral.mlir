// RUN: enzymexlamlir-opt --enzyme-hlo-opt --auto-batching --inline --enzyme-hlo-opt %s | FileCheck %s

module @reactant_updates2 attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<3x32x32xf32>, %arg1: tensor<3x32x32xf32>, %arg2: tensor<3x32x32xf32>) -> (tensor<3x32x32xf32>, tensor<3x32x32xf32>, tensor<3x32x32xf32>) {
    %0 = stablehlo.slice %arg0 [0:1, 0:32, 0:32] : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x32x32xf32>) -> tensor<32x32xf32>
    %2 = stablehlo.slice %arg1 [0:1, 0:32, 0:32] : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %3 = stablehlo.reshape %2 : (tensor<1x32x32xf32>) -> tensor<32x32xf32>
    %4 = stablehlo.dot_general %1, %3, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %5 = stablehlo.slice %arg2 [0:1, 0:32, 0:32] : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %6 = stablehlo.reshape %5 : (tensor<1x32x32xf32>) -> tensor<32x32xf32>
    %7 = stablehlo.dot_general %1, %6, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %8 = stablehlo.slice %arg0 [1:2, 0:32, 0:32] : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %9 = stablehlo.reshape %8 : (tensor<1x32x32xf32>) -> tensor<32x32xf32>
    %10 = stablehlo.slice %arg1 [1:2, 0:32, 0:32] : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %11 = stablehlo.reshape %10 : (tensor<1x32x32xf32>) -> tensor<32x32xf32>
    %12 = stablehlo.dot_general %9, %11, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %13 = stablehlo.slice %arg2 [1:2, 0:32, 0:32] : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %14 = stablehlo.reshape %13 : (tensor<1x32x32xf32>) -> tensor<32x32xf32>
    %15 = stablehlo.dot_general %9, %14, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %16 = stablehlo.slice %arg0 [2:3, 0:32, 0:32] : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %17 = stablehlo.reshape %16 : (tensor<1x32x32xf32>) -> tensor<32x32xf32>
    %18 = stablehlo.slice %arg1 [2:3, 0:32, 0:32] : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %19 = stablehlo.reshape %18 : (tensor<1x32x32xf32>) -> tensor<32x32xf32>
    %20 = stablehlo.dot_general %17, %19, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %21 = stablehlo.slice %arg2 [2:3, 0:32, 0:32] : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %22 = stablehlo.reshape %21 : (tensor<1x32x32xf32>) -> tensor<32x32xf32>
    %23 = stablehlo.dot_general %17, %22, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %24 = stablehlo.broadcast_in_dim %4, dims = [2, 1] : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
    %25 = stablehlo.broadcast_in_dim %12, dims = [2, 1] : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
    %26 = stablehlo.broadcast_in_dim %20, dims = [2, 1] : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
    %27 = stablehlo.broadcast_in_dim %7, dims = [2, 1] : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
    %28 = stablehlo.broadcast_in_dim %15, dims = [2, 1] : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
    %29 = stablehlo.broadcast_in_dim %23, dims = [2, 1] : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
    %30 = stablehlo.concatenate %24, %25, %26, dim = 0 : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<3x32x32xf32>
    %31 = stablehlo.concatenate %27, %28, %29, dim = 0 : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<3x32x32xf32>
    return %arg0, %30, %31 : tensor<3x32x32xf32>, tensor<3x32x32xf32>, tensor<3x32x32xf32>
  }
}

// CHECK: module @reactant_updates2 attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
// CHECK-NEXT:   func.func @main(%arg0: tensor<3x32x32xf32>, %arg1: tensor<3x32x32xf32>, %arg2: tensor<3x32x32xf32>) -> (tensor<3x32x32xf32>, tensor<3x32x32xf32>, tensor<3x32x32xf32>) {
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x32x32xf32>, tensor<3x32x32xf32>) -> tensor<3x32x32xf32>
// CHECK-NEXT:     %1 = stablehlo.dot_general %arg2, %arg0, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x32x32xf32>, tensor<3x32x32xf32>) -> tensor<3x32x32xf32>
// CHECK-NEXT:     return %arg0, %0, %1 : tensor<3x32x32xf32>, tensor<3x32x32xf32>, tensor<3x32x32xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
