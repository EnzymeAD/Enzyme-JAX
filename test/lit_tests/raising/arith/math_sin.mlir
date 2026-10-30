// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @sin_f32
  // CHECK: stablehlo.sine
  // CHECK-NOT: math.sin
  func.func @sin_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.sin %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @sin_f64
  // CHECK: stablehlo.sine
  // CHECK-NOT: math.sin
  func.func @sin_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.sin %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
