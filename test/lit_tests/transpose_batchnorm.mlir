// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=transpose_batch_norm_training;transpose_batch_norm_inference;transpose_batch_norm_grad},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<5x4x3x2xf64>, %arg1: tensor<3xf64>, %arg2: tensor<3xf64>) -> (tensor<5x4x3x2xf64>, tensor<3xf64>, tensor<3xf64>) {
    %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<5x4x3x2xf64>) -> tensor<2x3x4x5xf64>
    %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%0, %arg1, %arg2) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<2x3x4x5xf64>, tensor<3xf64>, tensor<3xf64>) -> (tensor<2x3x4x5xf64>, tensor<3xf64>, tensor<3xf64>)
    %1 = stablehlo.transpose %output, dims = [3, 2, 1, 0] : (tensor<2x3x4x5xf64>) -> tensor<5x4x3x2xf64>
    return %1, %batch_mean, %batch_var : tensor<5x4x3x2xf64>, tensor<3xf64>, tensor<3xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x4x3x2xf64>, %arg1: tensor<3xf64>, %arg2: tensor<3xf64>) -> (tensor<5x4x3x2xf64>, tensor<3xf64>, tensor<3xf64>) {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<5x4x3x2xf64>) -> tensor<2x3x4x5xf64>
// CHECK-NEXT:     %1 = stablehlo.transpose %0, dims = [3, 2, 1, 0] : (tensor<2x3x4x5xf64>) -> tensor<5x4x3x2xf64>
// CHECK-NEXT:     %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%1, %arg1, %arg2) <{epsilon = 9.99999974E-6 : f32, feature_index = 2 : i64}> : (tensor<5x4x3x2xf64>, tensor<3xf64>, tensor<3xf64>) -> (tensor<5x4x3x2xf64>, tensor<3xf64>, tensor<3xf64>)
// CHECK-NEXT:     return %output, %batch_mean, %batch_var : tensor<5x4x3x2xf64>, tensor<3xf64>, tensor<3xf64>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<5x4x3x2xf64>, %arg1: tensor<3xf64>, %arg2: tensor<3xf64>, %arg3: tensor<3xf64>, %arg4: tensor<3xf64>) -> tensor<5x4x3x2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<5x4x3x2xf64>) -> tensor<2x3x4x5xf64>
    %1 = "stablehlo.batch_norm_inference"(%0, %arg1, %arg2, %arg3, %arg4) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<2x3x4x5xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<2x3x4x5xf64>
    %2 = stablehlo.transpose %1, dims = [3, 2, 1, 0] : (tensor<2x3x4x5xf64>) -> tensor<5x4x3x2xf64>
    return %2 : tensor<5x4x3x2xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x4x3x2xf64>, %arg1: tensor<3xf64>, %arg2: tensor<3xf64>, %arg3: tensor<3xf64>, %arg4: tensor<3xf64>) -> tensor<5x4x3x2xf64> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<5x4x3x2xf64>) -> tensor<2x3x4x5xf64>
// CHECK-NEXT:     %1 = stablehlo.transpose %0, dims = [3, 2, 1, 0] : (tensor<2x3x4x5xf64>) -> tensor<5x4x3x2xf64>
// CHECK-NEXT:     %2 = "stablehlo.batch_norm_inference"(%1, %arg1, %arg2, %arg3, %arg4) <{epsilon = 9.99999974E-6 : f32, feature_index = 2 : i64}> : (tensor<5x4x3x2xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<5x4x3x2xf64>
// CHECK-NEXT:     return %2 : tensor<5x4x3x2xf64>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<5x4x3x2xf64>, %arg1: tensor<3xf64>, %arg2: tensor<3xf64>, %arg3: tensor<3xf64>, %arg4: tensor<5x4x3x2xf64>) -> (tensor<5x4x3x2xf64>, tensor<3xf64>, tensor<3xf64>) {
    %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<5x4x3x2xf64>) -> tensor<2x3x4x5xf64>
    %1 = stablehlo.transpose %arg4, dims = [3, 2, 1, 0] : (tensor<5x4x3x2xf64>) -> tensor<2x3x4x5xf64>
    %grad_operand, %grad_scale, %grad_offset = "stablehlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<2x3x4x5xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<2x3x4x5xf64>) -> (tensor<2x3x4x5xf64>, tensor<3xf64>, tensor<3xf64>)
    %2 = stablehlo.transpose %grad_operand, dims = [3, 2, 1, 0] : (tensor<2x3x4x5xf64>) -> tensor<5x4x3x2xf64>
    return %2, %grad_scale, %grad_offset : tensor<5x4x3x2xf64>, tensor<3xf64>, tensor<3xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x4x3x2xf64>, %arg1: tensor<3xf64>, %arg2: tensor<3xf64>, %arg3: tensor<3xf64>, %arg4: tensor<5x4x3x2xf64>) -> (tensor<5x4x3x2xf64>, tensor<3xf64>, tensor<3xf64>) {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<5x4x3x2xf64>) -> tensor<2x3x4x5xf64>
// CHECK-NEXT:     %1 = stablehlo.transpose %arg4, dims = [3, 2, 1, 0] : (tensor<5x4x3x2xf64>) -> tensor<2x3x4x5xf64>
// CHECK-NEXT:     %2 = stablehlo.transpose %0, dims = [3, 2, 1, 0] : (tensor<2x3x4x5xf64>) -> tensor<5x4x3x2xf64>
// CHECK-NEXT:     %3 = stablehlo.transpose %1, dims = [3, 2, 1, 0] : (tensor<2x3x4x5xf64>) -> tensor<5x4x3x2xf64>
// CHECK-NEXT:     %grad_operand, %grad_scale, %grad_offset = "stablehlo.batch_norm_grad"(%2, %arg1, %arg2, %arg3, %3) <{epsilon = 9.99999974E-6 : f32, feature_index = 2 : i64}> : (tensor<5x4x3x2xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<5x4x3x2xf64>) -> (tensor<5x4x3x2xf64>, tensor<3xf64>, tensor<3xf64>)
// CHECK-NEXT:     return %grad_operand, %grad_scale, %grad_offset : tensor<5x4x3x2xf64>, tensor<3xf64>, tensor<3xf64>
// CHECK-NEXT: }

module @"reactant_Chain{@..." attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<16x3x32x32xf32> {tf.aliasing_output = 3 : i32}, %arg1: tensor<3xf32> {tf.aliasing_output = 4 : i32}, %arg2: tensor<3xf32> {tf.aliasing_output = 5 : i32}, %arg3: tensor<3xf32> {tf.aliasing_output = 6 : i32}, %arg4: tensor<3xf32> {tf.aliasing_output = 7 : i32}) -> (tensor<16x3x1x1xf32>, tensor<3xf32>, tensor<3xf32>, tensor<16x3x32x32xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) {
    %c = stablehlo.constant dense<1024> : tensor<1x1x3x16xi64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.100006104> : tensor<3xf32>
    %cst_1 = stablehlo.constant dense<1.000000e-01> : tensor<3xf32>
    %cst_2 = stablehlo.constant dense<0.899999976> : tensor<3xf32>
    %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<16x3x32x32xf32>) -> tensor<32x32x3x16xf32>
    %1 = stablehlo.transpose %arg1, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %2 = stablehlo.transpose %arg2, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %3 = stablehlo.transpose %arg3, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %4 = stablehlo.transpose %arg4, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%0, %1, %2) <{epsilon = 9.99999974E-6 : f32, feature_index = 2 : i64}> : (tensor<32x32x3x16xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<32x32x3x16xf32>, tensor<3xf32>, tensor<3xf32>)
    %5 = stablehlo.multiply %cst_2, %3 : tensor<3xf32>
    %6 = stablehlo.multiply %cst_1, %batch_mean : tensor<3xf32>
    %7 = stablehlo.add %5, %6 : tensor<3xf32>
    %8 = stablehlo.transpose %7, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %9 = stablehlo.multiply %cst_2, %4 : tensor<3xf32>
    %10 = stablehlo.multiply %cst_0, %batch_var : tensor<3xf32>
    %11 = stablehlo.add %9, %10 : tensor<3xf32>
    %12 = stablehlo.transpose %11, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %13 = stablehlo.transpose %output, dims = [0, 1, 3, 2] : (tensor<32x32x3x16xf32>) -> tensor<32x32x16x3xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x32x16x3xf32>, tensor<f32>) -> tensor<16x3xf32>
    %15 = stablehlo.reshape %14 : (tensor<16x3xf32>) -> tensor<16x3x1x1xf32>
    %16 = stablehlo.transpose %15, dims = [3, 2, 1, 0] : (tensor<16x3x1x1xf32>) -> tensor<1x1x3x16xf32>
    %17 = stablehlo.convert %c : (tensor<1x1x3x16xi64>) -> tensor<1x1x3x16xf32>
    %18 = stablehlo.divide %16, %17 : tensor<1x1x3x16xf32>
    %19 = stablehlo.transpose %18, dims = [3, 2, 1, 0] : (tensor<1x1x3x16xf32>) -> tensor<16x3x1x1xf32>
    %20 = stablehlo.transpose %arg0, dims = [0, 1, 2, 3] : (tensor<16x3x32x32xf32>) -> tensor<16x3x32x32xf32>
    %21 = stablehlo.transpose %arg1, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %22 = stablehlo.transpose %arg2, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %23 = stablehlo.transpose %arg3, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %24 = stablehlo.transpose %arg4, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    return %19, %8, %12, %20, %21, %22, %23, %24 : tensor<16x3x1x1xf32>, tensor<3xf32>, tensor<3xf32>, tensor<16x3x32x32xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<16x3x32x32xf32> {tf.aliasing_output = 3 : i32}, %arg1: tensor<3xf32> {tf.aliasing_output = 4 : i32}, %arg2: tensor<3xf32> {tf.aliasing_output = 5 : i32}, %arg3: tensor<3xf32> {tf.aliasing_output = 6 : i32}, %arg4: tensor<3xf32> {tf.aliasing_output = 7 : i32}) -> (tensor<16x3x1x1xf32>, tensor<3xf32>, tensor<3xf32>, tensor<16x3x32x32xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) {
// CHECK-NEXT:     %c = stablehlo.constant dense<1024> : tensor<1x1x3x16xi64>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.100006104> : tensor<3xf32>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<1.000000e-01> : tensor<3xf32>
// CHECK-NEXT:     %cst_2 = stablehlo.constant dense<0.899999976> : tensor<3xf32>
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<16x3x32x32xf32>) -> tensor<32x32x3x16xf32>
// CHECK-NEXT:     %1 = stablehlo.transpose %arg1, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
// CHECK-NEXT:     %2 = stablehlo.transpose %arg2, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
// CHECK-NEXT:     %3 = stablehlo.transpose %arg3, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
// CHECK-NEXT:     %4 = stablehlo.transpose %arg4, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
// CHECK-NEXT:     %5 = stablehlo.transpose %0, dims = [0, 1, 3, 2] : (tensor<32x32x3x16xf32>) -> tensor<32x32x16x3xf32>
// CHECK-NEXT:     %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%5, %1, %2) <{epsilon = 9.99999974E-6 : f32, feature_index = 3 : i64}> : (tensor<32x32x16x3xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<32x32x16x3xf32>, tensor<3xf32>, tensor<3xf32>)
// CHECK-NEXT:     %6 = stablehlo.multiply %cst_2, %3 : tensor<3xf32>
// CHECK-NEXT:     %7 = stablehlo.multiply %cst_1, %batch_mean : tensor<3xf32>
// CHECK-NEXT:     %8 = stablehlo.add %6, %7 : tensor<3xf32>
// CHECK-NEXT:     %9 = stablehlo.transpose %8, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
// CHECK-NEXT:     %10 = stablehlo.multiply %cst_2, %4 : tensor<3xf32>
// CHECK-NEXT:     %11 = stablehlo.multiply %cst_0, %batch_var : tensor<3xf32>
// CHECK-NEXT:     %12 = stablehlo.add %10, %11 : tensor<3xf32>
// CHECK-NEXT:     %13 = stablehlo.transpose %12, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
// CHECK-NEXT:     %14 = stablehlo.reduce(%output init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x32x16x3xf32>, tensor<f32>) -> tensor<16x3xf32>
// CHECK-NEXT:     %15 = stablehlo.reshape %14 : (tensor<16x3xf32>) -> tensor<16x3x1x1xf32>
// CHECK-NEXT:     %16 = stablehlo.transpose %15, dims = [3, 2, 1, 0] : (tensor<16x3x1x1xf32>) -> tensor<1x1x3x16xf32>
// CHECK-NEXT:     %17 = stablehlo.convert %c : (tensor<1x1x3x16xi64>) -> tensor<1x1x3x16xf32>
// CHECK-NEXT:     %18 = stablehlo.divide %16, %17 : tensor<1x1x3x16xf32>
// CHECK-NEXT:     %19 = stablehlo.transpose %18, dims = [3, 2, 1, 0] : (tensor<1x1x3x16xf32>) -> tensor<16x3x1x1xf32>
// CHECK-NEXT:     %20 = stablehlo.transpose %arg0, dims = [0, 1, 2, 3] : (tensor<16x3x32x32xf32>) -> tensor<16x3x32x32xf32>
// CHECK-NEXT:     %21 = stablehlo.transpose %arg1, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
// CHECK-NEXT:     %22 = stablehlo.transpose %arg2, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
// CHECK-NEXT:     %23 = stablehlo.transpose %arg3, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
// CHECK-NEXT:     %24 = stablehlo.transpose %arg4, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
// CHECK-NEXT:     return %19, %9, %13, %20, %21, %22, %23, %24 : tensor<16x3x1x1xf32>, tensor<3xf32>, tensor<3xf32>, tensor<16x3x32x32xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:   }
