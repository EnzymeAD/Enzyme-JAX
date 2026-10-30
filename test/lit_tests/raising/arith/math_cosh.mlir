// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @cosh_f32
  // CHECK: chlo.cosh
  // CHECK-NOT: math.cosh
  func.func @cosh_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.cosh %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @cosh_f64
  // CHECK: chlo.cosh
  // CHECK-NOT: math.cosh
  func.func @cosh_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.cosh %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
