// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=512})" %s | FileCheck %s

module {
  func.func private @transpose_reduce(%in: tensor<9x20x45xf64>) -> (tensor<45x20xf64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %reduce = stablehlo.reduce(%in init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<9x20x45xf64>, tensor<f64>) -> tensor<20x45xf64>
    %transpose = "stablehlo.transpose"(%reduce) { permutation = array<i64 : 1, 0> } : (tensor<20x45xf64>) -> tensor<45x20xf64>
    return %transpose : tensor<45x20xf64>
  }
}

// CHECK: func.func private @transpose_reduce(%arg0: tensor<9x20x45xf64>) -> tensor<45x20xf64> {
// CHECK:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:   %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<9x20x45xf64>) -> tensor<9x45x20xf64>
// CHECK:   %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<9x20x45xf64>, tensor<f64>) -> tensor<20x45xf64>
// CHECK:   %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<20x45xf64>) -> tensor<45x20xf64>
// CHECK:   return %2 : tensor<45x20xf64>
// CHECK: }
