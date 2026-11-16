// RUN: enzymexlamlir-opt %s "--enzyme-hlo-generate-td=patterns=full_reduce_reshape_or_transpose" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

func.func @main(%g: tensor<2x2x2xf64>, %w: tensor<2x2x2xf64>) -> tensor<f64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %a = stablehlo.transpose %g, dims = [0, 2, 1] : (tensor<2x2x2xf64>) -> tensor<2x2x2xf64>
  %b = stablehlo.transpose %w, dims = [2, 1, 0] : (tensor<2x2x2xf64>) -> tensor<2x2x2xf64>
  %m = stablehlo.multiply %a, %b : tensor<2x2x2xf64>
  %n = stablehlo.transpose %m, dims = [0, 2, 1] : (tensor<2x2x2xf64>) -> tensor<2x2x2xf64>
  %r = stablehlo.reduce(%n init: %cst) applies stablehlo.add across dimensions = [0, 1, 2]
      : (tensor<2x2x2xf64>, tensor<f64>) -> tensor<f64>
  return %r : tensor<f64>
}

// CHECK:  func.func @main(%arg0: tensor<2x2x2xf64>, %arg1: tensor<2x2x2xf64>) -> tensor<f64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<2x2x2xf64>) -> tensor<2x2x2xf64>
// CHECK-NEXT:    %1 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<2x2x2xf64>) -> tensor<2x2x2xf64>
// CHECK-NEXT:    %2 = stablehlo.multiply %0, %1 : tensor<2x2x2xf64>
// CHECK-NEXT:    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<2x2x2xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    return %3 : tensor<f64>
// CHECK-NEXT:  }
