// RUN: enzymexlamlir-opt %s -parse-optimization-rules -llvm-to-tessera -tessera-apply-pdl -tessera-to-func | FileCheck %s

// Tessera on func.func and tensors, as emitted by Reactant.jl. The literal in
// the pow rule matches a splat tensor constant.
module {
  tessera.optimizations {
    tessera.optimization "tessera.inv(tessera.inv(x)) -> x"
    tessera.optimization "tessera.pow(x, 2) -> tessera.mul(x, x)"
  }
  func.func private @inv(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {pure_tessera_op = "tessera.inv(x)"} {
    %c = stablehlo.constant dense<1.0> : tensor<4xf32>
    %0 = stablehlo.divide %c, %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
  func.func private @pow(%arg0: tensor<4xf32>, %arg1: tensor<i64>) -> tensor<4xf32> attributes {pure_tessera_op = "tessera.pow(x, n)"} {
    return %arg0 : tensor<4xf32>
  }
  func.func private @mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> attributes {pure_tessera_op = "tessera.mul(x, y)"} {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %0 = call @inv(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
    %1 = call @inv(%0) : (tensor<4xf32>) -> tensor<4xf32>
    %c = stablehlo.constant dense<2> : tensor<i64>
    %2 = call @pow(%1, %c) : (tensor<4xf32>, tensor<i64>) -> tensor<4xf32>
    return %1, %2 : tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK: func.func private @inv(
// CHECK-NOT: tessera
// CHECK: func.func @main(%[[X:.*]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT: %[[M:.*]] = call @mul(%[[X]], %[[X]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[X]], %[[M]] : tensor<4xf32>, tensor<4xf32>
