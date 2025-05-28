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
