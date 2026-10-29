// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @acosh_f32
  // CHECK: %[[ACOSH:.+]] = chlo.acosh %arg0 : tensor<4xf32>
  // CHECK: return %[[ACOSH]] : tensor<4xf32>
  // CHECK-NOT: math.acosh
  func.func @acosh_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.acosh %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @acosh_f64
  // CHECK: %[[ACOSH:.+]] = chlo.acosh %arg0 : tensor<4xf64>
  // CHECK: return %[[ACOSH]] : tensor<4xf64>
  // CHECK-NOT: math.acosh
  func.func @acosh_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.acosh %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
