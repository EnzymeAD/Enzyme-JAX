// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @asin_f32
  // CHECK: %[[ASIN:.+]] = chlo.asin %arg0 : tensor<4xf32>
  // CHECK: return %[[ASIN]] : tensor<4xf32>
  // CHECK-NOT: math.asin
  func.func @asin_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.asin %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @asin_f64
  // CHECK: %[[ASIN:.+]] = chlo.asin %arg0 : tensor<4xf64>
  // CHECK: return %[[ASIN]] : tensor<4xf64>
  // CHECK-NOT: math.asin
  func.func @asin_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.asin %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
