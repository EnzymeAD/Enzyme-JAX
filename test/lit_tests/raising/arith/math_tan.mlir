// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @tan_f32
  // CHECK: stablehlo.tan
  // CHECK-NOT: math.tan
  func.func @tan_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.tan %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @tan_f64
  // CHECK: stablehlo.tan
  // CHECK-NOT: math.tan
  func.func @tan_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.tan %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
