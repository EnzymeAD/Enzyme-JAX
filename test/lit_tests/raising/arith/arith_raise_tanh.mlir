// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @tanh_f32
  // CHECK: stablehlo.tanh
  // CHECK-NOT: math.tanh
  func.func @tanh_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.tanh %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @tanh_f64
  // CHECK: stablehlo.tanh
  // CHECK-NOT: math.tanh
  func.func @tanh_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.tanh %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
