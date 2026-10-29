// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @atanh_f32
  // CHECK: %[[ATANH:.+]] = chlo.atanh %arg0 : tensor<4xf32>
  // CHECK: return %[[ATANH]] : tensor<4xf32>
  // CHECK-NOT: math.atanh
  func.func @atanh_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.atanh %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @atanh_f64
  // CHECK: %[[ATANH:.+]] = chlo.atanh %arg0 : tensor<4xf64>
  // CHECK: return %[[ATANH]] : tensor<4xf64>
  // CHECK-NOT: math.atanh
  func.func @atanh_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.atanh %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
