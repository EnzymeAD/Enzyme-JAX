// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @sinh_f32
  // CHECK: chlo.sinh
  // CHECK-NOT: math.sinh
  func.func @sinh_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.sinh %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @sinh_f64
  // CHECK: chlo.sinh
  // CHECK-NOT: math.sinh
  func.func @sinh_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.sinh %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
