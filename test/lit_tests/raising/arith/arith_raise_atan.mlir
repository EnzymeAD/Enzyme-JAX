// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @atan_f32
  // CHECK: %[[ATAN:.+]] = chlo.atan %arg0 : tensor<4xf32>
  // CHECK: return %[[ATAN]] : tensor<4xf32>
  // CHECK-NOT: math.atan
  func.func @atan_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.atan %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @atan_f64
  // CHECK: %[[ATAN:.+]] = chlo.atan %arg0 : tensor<4xf64>
  // CHECK: return %[[ATAN]] : tensor<4xf64>
  // CHECK-NOT: math.atan
  func.func @atan_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.atan %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
