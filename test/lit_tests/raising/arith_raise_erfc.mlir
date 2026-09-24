// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @erfc_f32
  // CHECK: %[[ERFC:.+]] = chlo.erfc %arg0 : tensor<4xf32>
  // CHECK: return %[[ERFC]] : tensor<4xf32>
  // CHECK-NOT: math.erfc
  func.func @erfc_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.erfc %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @erfc_f64
  // CHECK: %[[ERFC:.+]] = chlo.erfc %arg0 : tensor<4xf64>
  // CHECK: return %[[ERFC]] : tensor<4xf64>
  // CHECK-NOT: math.erfc
  func.func @erfc_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.erfc %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
