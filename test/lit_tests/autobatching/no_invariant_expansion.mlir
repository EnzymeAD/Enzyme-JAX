// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s --implicit-check-not="tensor<30x8x4xf64>"

// #3965: w(k) depends only on the induction variable and %d is loop invariant,
// so w(k) * %d can be computed for every iteration before the loop. That would
// materialize a 30x8x4 array (growing with the trip count) to replace a
// per-iteration product with an 8x4 operand, so the product stays in the loop.

module {
  func.func @main(%x0: tensor<8x4xf64>, %d: tensor<8x4xf64>) -> tensor<8x4xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c10 = stablehlo.constant dense<10> : tensor<i64>
    %c30 = stablehlo.constant dense<30> : tensor<i64>
    %one = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %eps = stablehlo.constant dense<1.000000e-02> : tensor<8x4xf64>
    %r:2 = stablehlo.while(%k = %c0, %x = %x0) : tensor<i64>, tensor<8x4xf64>
    cond {
      %lt = stablehlo.compare LT, %k, %c30 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %lt : tensor<i1>
    } do {
      %kn = stablehlo.add %k, %c1 : tensor<i64>
      %le = stablehlo.compare LE, %kn, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %w = stablehlo.select %le, %one, %zero : tensor<i1>, tensor<f64>
      %wb = stablehlo.broadcast_in_dim %w, dims = [] : (tensor<f64>) -> tensor<8x4xf64>
      %wd = stablehlo.multiply %wb, %d : tensor<8x4xf64>
      %xx = stablehlo.multiply %x, %x : tensor<8x4xf64>
      %s = stablehlo.subtract %wd, %xx : tensor<8x4xf64>
      %t = stablehlo.multiply %eps, %s : tensor<8x4xf64>
      %xn = stablehlo.add %x, %t : tensor<8x4xf64>
      stablehlo.return %kn, %xn : tensor<i64>, tensor<8x4xf64>
    }
    return %r#1 : tensor<8x4xf64>
  }
}

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// CHECK: stablehlo.multiply
