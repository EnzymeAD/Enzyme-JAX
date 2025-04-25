// RUN: enzymexlamlir-opt %s --enzyme --arith-raise --enzyme-hlo-opt | FileCheck %s

module @reactant_vector_... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"Const{typeof(fn)}(Main.fn)_autodiff"(%arg0: tensor<2x2xf32>) -> (tensor<f32>, tensor<2x2xf32>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<2x2xf32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<f32>
    return %2, %arg0 : tensor<f32>, tensor<2x2xf32>
  }
  func.func @main(%arg0: tensor<2x2xf32> {tf.aliasing_output = 4 : i32}) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<2x2xf32>) {
    %cst = stablehlo.constant dense<[[[1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 1.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [1.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]]]> : tensor<4x2x2xf32>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1:3 = enzyme.fwddiff @"Const{typeof(fn)}(Main.fn)_autodiff"(%0, %cst) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>, #enzyme<activity enzyme_dup>], width = 4 : i64} : (tensor<2x2xf32>, tensor<4x2x2xf32>) -> (tensor<4xf32>, tensor<2x2xf32>, tensor<4x2x2xf32>)
    %2 = stablehlo.slice %1#0 [0:1] : (tensor<4xf32>) -> tensor<1xf32>
    %3 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.slice %1#0 [1:2] : (tensor<4xf32>) -> tensor<1xf32>
    %5 = stablehlo.reshape %4 : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.slice %1#0 [2:3] : (tensor<4xf32>) -> tensor<1xf32>
    %7 = stablehlo.reshape %6 : (tensor<1xf32>) -> tensor<f32>
    %8 = stablehlo.slice %1#0 [3:4] : (tensor<4xf32>) -> tensor<1xf32>
    %9 = stablehlo.reshape %8 : (tensor<1xf32>) -> tensor<f32>
    return %3, %5, %7, %9, %1#1 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<2x2xf32>
  }
}

// CHECK:  func.func private @"fwddiffe4Const{typeof(fn)}(Main.fn)_autodiff"(%arg0: tensor<2x2xf32>, %arg1: tensor<4x2x2xf32>) -> (tensor<4xf32>, tensor<2x2xf32>, tensor<4x2x2xf32>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.transpose %arg1, dims = [0, 2, 1] : (tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %arg0, dims = [2, 1] : (tensor<2x2xf32>) -> tensor<4x2x2xf32>
// CHECK-NEXT:    %2 = stablehlo.multiply %0, %1 : tensor<4x2x2xf32>
// CHECK-NEXT:    %3 = stablehlo.add %2, %2 : tensor<4x2x2xf32>
// CHECK-NEXT:    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<4x2x2xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT:    return %4, %arg0, %arg1 : tensor<4xf32>, tensor<2x2xf32>, tensor<4x2x2xf32>
// CHECK-NEXT:  }
