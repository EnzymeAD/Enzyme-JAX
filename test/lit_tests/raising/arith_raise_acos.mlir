// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @acos_f32
  // CHECK: %[[ACOS:.+]] = chlo.acos %arg0 : tensor<4xf32>
  // CHECK: return %[[ACOS]] : tensor<4xf32>
  // CHECK-NOT: math.acos
  func.func @acos_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.acos %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @acos_f64
  // CHECK: %[[ACOS:.+]] = chlo.acos %arg0 : tensor<4xf64>
  // CHECK: return %[[ACOS]] : tensor<4xf64>
  // CHECK-NOT: math.acos
  func.func @acos_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.acos %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
