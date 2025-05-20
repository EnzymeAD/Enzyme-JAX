// RUN: enzymexlamlir-opt %s --enzyme-batch | FileCheck %s

func.func @main(%arg0: tensor<4x16xf64>, %arg1: tensor<4x5xf64>) -> tensor<4x16xf64> {
    %0 = enzyme.batch @dus(%arg0, %arg1) {batch_shape = array<i64: 4>} : (tensor<4x16xf64>, tensor<4x5xf64>) -> tensor<4x16xf64>
    return %0 : tensor<4x16xf64>
}

func.func @dus(%arg0: tensor<16xf64>, %arg1: tensor<5xf64>) -> tensor<16xf64> {
    %c = stablehlo.constant dense<6> : tensor<i64>
    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c : (tensor<16xf64>, tensor<5xf64>, tensor<i64>) -> tensor<16xf64>
    return %0 : tensor<16xf64>
}

// CHECK: func.func private @batched_dus(%arg0: tensor<4x16xf64>, %arg1: tensor<4x5xf64>) -> tensor<4x16xf64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<6> : tensor<4xi64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %0 = stablehlo.slice %c [0:1] : (tensor<4xi64>) -> tensor<1xi64>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:     %2 = stablehlo.dynamic_update_slice %arg0, %arg1, %c_0, %1 : (tensor<4x16xf64>, tensor<4x5xf64>, tensor<i64>, tensor<i64>) -> tensor<4x16xf64>
// CHECK-NEXT:     return %2 : tensor<4x16xf64>
// CHECK-NEXT: }
