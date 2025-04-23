// RUN: enzymexlamlir-opt %s --enzyme-batch --enzyme-hlo-opt | FileCheck %s

func.func private @sum(%arg0: tensor<16xf64>) -> (tensor<f64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16xf64>, tensor<f64>) -> tensor<f64>
    return %1 : tensor<f64>
}
func.func @main(%arg0: tensor<4x16xf64>) -> (tensor<4xf64>) {
    %1 = enzyme.batch @sum(%arg0) {batch_shape = array<i64: 4>} : (tensor<4x16xf64>) -> (tensor<4xf64>)
    return %1 : tensor<4xf64>
}

// CHECK: func.func @main(%arg0: tensor<4x16xf64>) -> tensor<4xf64> {
// CHECK-NEXT:    %0 = call @batched_sum(%arg0) : (tensor<4x16xf64>) -> tensor<4xf64>
// CHECK-NEXT:    return %0 : tensor<4xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @batched_sum(%arg0: tensor<4x16xf64>) -> tensor<4xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x16xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:    return %0 : tensor<4xf64>
// CHECK-NEXT:  }
