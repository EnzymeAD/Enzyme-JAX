// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_reduce" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func private @transpose_reduce(%in: tensor<9x20x45xf64>) -> (tensor<45x20xf64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %reduce = stablehlo.reduce(%in init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<9x20x45xf64>, tensor<f64>) -> tensor<20x45xf64>
    %transpose = "stablehlo.transpose"(%reduce) { permutation = array<i64 : 1, 0> } : (tensor<20x45xf64>) -> tensor<45x20xf64>
    return %transpose : tensor<45x20xf64>
  }
}

// CHECK:  func.func private @transpose_reduce(%arg0: tensor<9x20x45xf64>) -> tensor<45x20xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<9x20x45xf64>) -> tensor<9x45x20xf64>
// CHECK-NEXT:    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<9x45x20xf64>, tensor<f64>) -> tensor<45x20xf64>
// CHECK-NEXT:    return %1 : tensor<45x20xf64>
// CHECK-NEXT:  }
