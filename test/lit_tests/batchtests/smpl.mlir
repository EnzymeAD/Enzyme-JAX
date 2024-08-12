// RUN: enzymexlamlir-opt %s --enzyme-batch | FileCheck %s --check-prefix=FORWARD
  
func.func private @relu_broadcast_scalar(%arg0: tensor<f64>) -> (tensor<f64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = stablehlo.maximum %arg0, %cst : tensor<f64>
    return %1 : tensor<f64>
  }
  func.func @main(%arg0: tensor<2x2xf64>) -> (tensor<2x2xf64>) {
    %1 = enzyme.batch @relu_broadcast_scalar(%arg0) {batch_shape = array<i64: 2, 2>} : (tensor<2x2xf64>) -> (tensor<2x2xf64>)
    return %1 : tensor<2x2xf64>
  }

// CHECK:  func.func @main(%arg0: tensor<2x2xf64>) -> tensor<2x2xf64> {
// CHECK-NEXT:    %0 = call @batched_relu_broadcast_scalar(%arg0) : (tensor<2x2xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:    return %0 : tensor<2x2xf64>
// CHECK-NEXT:  }
// CHECK:  func.func private @batched_relu_broadcast_scalar(%arg0: tensor<2x2xf64>) -> tensor<2x2xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf64>
// CHECK-NEXT:    %0 = stablehlo.maximum %arg0, %cst : tensor<2x2xf64>
// CHECK-NEXT:    return %0 : tensor<2x2xf64>
// CHECK-NEXT:  }
