// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=concat_insert_dim_dot_general},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module @reactant_batched... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<5x4x3xf64>, %arg1: tensor<7x4xf64>) -> tensor<5x7x3xf64> {
    %0 = stablehlo.slice %arg0 [4:5, 0:4, 0:3] : (tensor<5x4x3xf64>) -> tensor<1x4x3xf64>
    %1 = stablehlo.slice %arg0 [3:4, 0:4, 0:3] : (tensor<5x4x3xf64>) -> tensor<1x4x3xf64>
    %2 = stablehlo.slice %arg0 [2:3, 0:4, 0:3] : (tensor<5x4x3xf64>) -> tensor<1x4x3xf64>
    %3 = stablehlo.slice %arg0 [1:2, 0:4, 0:3] : (tensor<5x4x3xf64>) -> tensor<1x4x3xf64>
    %4 = stablehlo.slice %arg0 [0:1, 0:4, 0:3] : (tensor<5x4x3xf64>) -> tensor<1x4x3xf64>
    %5 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<7x4xf64>) -> tensor<4x7xf64>
    %6 = stablehlo.reshape %4 : (tensor<1x4x3xf64>) -> tensor<4x3xf64>
    %7 = stablehlo.transpose %6, dims = [1, 0] : (tensor<4x3xf64>) -> tensor<3x4xf64>
    %8 = stablehlo.dot_general %7, %5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<4x7xf64>) -> tensor<3x7xf64>
    %9 = stablehlo.reshape %3 : (tensor<1x4x3xf64>) -> tensor<4x3xf64>
    %10 = stablehlo.transpose %9, dims = [1, 0] : (tensor<4x3xf64>) -> tensor<3x4xf64>
    %11 = stablehlo.dot_general %10, %5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<4x7xf64>) -> tensor<3x7xf64>
    %12 = stablehlo.reshape %2 : (tensor<1x4x3xf64>) -> tensor<4x3xf64>
    %13 = stablehlo.transpose %12, dims = [1, 0] : (tensor<4x3xf64>) -> tensor<3x4xf64>
    %14 = stablehlo.dot_general %13, %5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<4x7xf64>) -> tensor<3x7xf64>
    %15 = stablehlo.reshape %1 : (tensor<1x4x3xf64>) -> tensor<4x3xf64>
    %16 = stablehlo.transpose %15, dims = [1, 0] : (tensor<4x3xf64>) -> tensor<3x4xf64>
    %17 = stablehlo.dot_general %16, %5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<4x7xf64>) -> tensor<3x7xf64>
    %18 = stablehlo.reshape %0 : (tensor<1x4x3xf64>) -> tensor<4x3xf64>
    %19 = stablehlo.transpose %18, dims = [1, 0] : (tensor<4x3xf64>) -> tensor<3x4xf64>
    %20 = stablehlo.dot_general %19, %5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<4x7xf64>) -> tensor<3x7xf64>
    %21 = stablehlo.reshape %8 : (tensor<3x7xf64>) -> tensor<3x7x1xf64>
    %22 = stablehlo.reshape %11 : (tensor<3x7xf64>) -> tensor<3x7x1xf64>
    %23 = stablehlo.reshape %14 : (tensor<3x7xf64>) -> tensor<3x7x1xf64>
    %24 = stablehlo.reshape %17 : (tensor<3x7xf64>) -> tensor<3x7x1xf64>
    %25 = stablehlo.reshape %20 : (tensor<3x7xf64>) -> tensor<3x7x1xf64>
    %26 = stablehlo.concatenate %21, %22, %23, %24, %25, dim = 2 : (tensor<3x7x1xf64>, tensor<3x7x1xf64>, tensor<3x7x1xf64>, tensor<3x7x1xf64>, tensor<3x7x1xf64>) -> tensor<3x7x5xf64>
    %27 = stablehlo.transpose %26, dims = [2, 1, 0] : (tensor<3x7x5xf64>) -> tensor<5x7x3xf64>
    return %27 : tensor<5x7x3xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x4x3xf64>, %arg1: tensor<7x4xf64>) -> tensor<5x7x3xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [4:5, 0:4, 0:3] : (tensor<5x4x3xf64>) -> tensor<1x4x3xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [3:4, 0:4, 0:3] : (tensor<5x4x3xf64>) -> tensor<1x4x3xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [2:3, 0:4, 0:3] : (tensor<5x4x3xf64>) -> tensor<1x4x3xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %arg0 [1:2, 0:4, 0:3] : (tensor<5x4x3xf64>) -> tensor<1x4x3xf64>
// CHECK-NEXT:    %4 = stablehlo.slice %arg0 [0:1, 0:4, 0:3] : (tensor<5x4x3xf64>) -> tensor<1x4x3xf64>
// CHECK-NEXT:    %5 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<7x4xf64>) -> tensor<4x7xf64>
// CHECK-NEXT:    %6 = stablehlo.reshape %4 : (tensor<1x4x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %7 = stablehlo.transpose %6, dims = [1, 0] : (tensor<4x3xf64>) -> tensor<3x4xf64>
// CHECK-NEXT:    %8 = stablehlo.reshape %3 : (tensor<1x4x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %9 = stablehlo.transpose %8, dims = [1, 0] : (tensor<4x3xf64>) -> tensor<3x4xf64>
// CHECK-NEXT:    %10 = stablehlo.reshape %2 : (tensor<1x4x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %11 = stablehlo.transpose %10, dims = [1, 0] : (tensor<4x3xf64>) -> tensor<3x4xf64>
// CHECK-NEXT:    %12 = stablehlo.reshape %1 : (tensor<1x4x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %13 = stablehlo.transpose %12, dims = [1, 0] : (tensor<4x3xf64>) -> tensor<3x4xf64>
// CHECK-NEXT:    %14 = stablehlo.reshape %0 : (tensor<1x4x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %15 = stablehlo.transpose %14, dims = [1, 0] : (tensor<4x3xf64>) -> tensor<3x4xf64>
// CHECK-NEXT:    %16 = stablehlo.reshape %7 : (tensor<3x4xf64>) -> tensor<1x3x4xf64>
// CHECK-NEXT:    %17 = stablehlo.reshape %9 : (tensor<3x4xf64>) -> tensor<1x3x4xf64>
// CHECK-NEXT:    %18 = stablehlo.reshape %11 : (tensor<3x4xf64>) -> tensor<1x3x4xf64>
// CHECK-NEXT:    %19 = stablehlo.reshape %13 : (tensor<3x4xf64>) -> tensor<1x3x4xf64>
// CHECK-NEXT:    %20 = stablehlo.reshape %15 : (tensor<3x4xf64>) -> tensor<1x3x4xf64>
// CHECK-NEXT:    %21 = stablehlo.concatenate %16, %17, %18, %19, %20, dim = 0 : (tensor<1x3x4xf64>, tensor<1x3x4xf64>, tensor<1x3x4xf64>, tensor<1x3x4xf64>, tensor<1x3x4xf64>) -> tensor<5x3x4xf64>
// CHECK-NEXT:    %22 = stablehlo.reshape %5 : (tensor<4x7xf64>) -> tensor<1x4x7xf64>
// CHECK-NEXT:    %23 = stablehlo.reshape %5 : (tensor<4x7xf64>) -> tensor<1x4x7xf64>
// CHECK-NEXT:    %24 = stablehlo.reshape %5 : (tensor<4x7xf64>) -> tensor<1x4x7xf64>
// CHECK-NEXT:    %25 = stablehlo.reshape %5 : (tensor<4x7xf64>) -> tensor<1x4x7xf64>
// CHECK-NEXT:    %26 = stablehlo.reshape %5 : (tensor<4x7xf64>) -> tensor<1x4x7xf64>
// CHECK-NEXT:    %27 = stablehlo.concatenate %22, %23, %24, %25, %26, dim = 0 : (tensor<1x4x7xf64>, tensor<1x4x7xf64>, tensor<1x4x7xf64>, tensor<1x4x7xf64>, tensor<1x4x7xf64>) -> tensor<5x4x7xf64>
// CHECK-NEXT:    %28 = call [[FN_NAME:@.*]](%21, %27) : (tensor<5x3x4xf64>, tensor<5x4x7xf64>) -> tensor<5x3x7xf64>
// CHECK-NEXT:    %29 = stablehlo.transpose %28, dims = [1, 2, 0] : (tensor<5x3x7xf64>) -> tensor<3x7x5xf64>
// CHECK-NEXT:    %30 = stablehlo.transpose %29, dims = [2, 1, 0] : (tensor<3x7x5xf64>) -> tensor<5x7x3xf64>
// CHECK-NEXT:    return %30 : tensor<5x7x3xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private [[FN_NAME]](%arg0: tensor<5x3x4xf64>, %arg1: tensor<5x4x7xf64>) -> tensor<5x3x7xf64> {
// CHECK-NEXT:    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x3x4xf64>, tensor<5x4x7xf64>) -> tensor<5x3x7xf64>
// CHECK-NEXT:    return %0 : tensor<5x3x7xf64>
// CHECK-NEXT:  }
