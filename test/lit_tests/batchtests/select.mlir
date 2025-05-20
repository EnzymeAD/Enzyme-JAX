// RUN: enzymexlamlir-opt --enzyme-batch %s | FileCheck %s

func.func private @select_1(%arg0: tensor<i1>, %arg1: tensor<3x8xf64>, %arg2: tensor<3x8xf64>) -> tensor<3x8xf64> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : (tensor<i1>, tensor<3x8xf64>, tensor<3x8xf64>) -> tensor<3x8xf64>
    return %0 : tensor<3x8xf64>
}

func.func @select_2(%arg0: tensor<3x8xi1>, %arg1: tensor<3x8xf64>, %arg2: tensor<3x8xf64>) -> tensor<3x8xf64> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : (tensor<3x8xi1>, tensor<3x8xf64>, tensor<3x8xf64>) -> tensor<3x8xf64>
    return %0 : tensor<3x8xf64>
}

func.func @main(%arg0: tensor<2x5xi1>, %arg1: tensor<2x5x3x8xi1>, %arg2: tensor<2x5x3x8xf64>, %arg3: tensor<2x5x3x8xf64>) -> tensor<2x5x3x8xf64> {
    %c = stablehlo.constant dense<1.000000e+00> : tensor<2x5x3x8xf64>
    %0 = enzyme.batch @select_1(%arg0, %arg2, %c) {batch_shape = array<i64: 2, 5>} : (tensor<2x5xi1>, tensor<2x5x3x8xf64>, tensor<2x5x3x8xf64>) -> (tensor<2x5x3x8xf64>)
    %1 = enzyme.batch @select_2(%arg1, %arg2, %c) {batch_shape = array<i64: 2, 5>} : (tensor<2x5x3x8xi1>, tensor<2x5x3x8xf64>, tensor<2x5x3x8xf64>) -> (tensor<2x5x3x8xf64>)
    %2 = stablehlo.add %0, %1 : tensor<2x5x3x8xf64>
    return %2 : tensor<2x5x3x8xf64>
}

// CHECK: func.func private @batched_select_1(%arg0: tensor<2x5xi1>, %arg1: tensor<2x5x3x8xf64>, %arg2: tensor<2x5x3x8xf64>) -> tensor<2x5x3x8xf64> {
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x5xi1>) -> tensor<2x5x3x8xi1>
// CHECK-NEXT:     %1 = stablehlo.select %0, %arg1, %arg2 : tensor<2x5x3x8xi1>, tensor<2x5x3x8xf64>
// CHECK-NEXT:     return %1 : tensor<2x5x3x8xf64>
// CHECK-NEXT: }
// CHECK: func.func private @batched_select_2(%arg0: tensor<2x5x3x8xi1>, %arg1: tensor<2x5x3x8xf64>, %arg2: tensor<2x5x3x8xf64>) -> tensor<2x5x3x8xf64> {
// CHECK-NEXT:     %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<2x5x3x8xi1>, tensor<2x5x3x8xf64>
// CHECK-NEXT:     return %0 : tensor<2x5x3x8xf64>
// CHECK-NEXT: }
