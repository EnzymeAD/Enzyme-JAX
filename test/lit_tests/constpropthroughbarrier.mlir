// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @transpose(%a : tensor<1xf32>, %b : tensor<1xf32>) -> (tensor<1xf32>, tensor<i64>) {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %r:3 = stablehlo.optimization_barrier %a, %b, %c : tensor<1xf32>, tensor<1xf32>, tensor<i64>
  return %r#0, %r#2 : tensor<1xf32>, tensor<i64>
}

// CHECK:  func.func @transpose(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> (tensor<1xf32>, tensor<i64>) {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.optimization_barrier %arg0 : tensor<1xf32>
// CHECK-NEXT:    return %[[i1]], %[[i0]] : tensor<1xf32>, tensor<i64>
// CHECK-NEXT:  }
