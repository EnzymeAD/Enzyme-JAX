// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @asinh_f32
  // CHECK: chlo.asinh
  // CHECK-NOT: math.asinh
  func.func @asinh_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.asinh %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @asinh_f64
  // CHECK: chlo.asinh
  // CHECK-NOT: math.asinh
  func.func @asinh_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.asinh %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
