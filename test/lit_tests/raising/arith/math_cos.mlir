// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @cos_f32
  // CHECK: stablehlo.cosine
  // CHECK-NOT: math.cos
  func.func @cos_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.cos %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @cos_f64
  // CHECK: stablehlo.cosine
  // CHECK-NOT: math.cos
  func.func @cos_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.cos %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
