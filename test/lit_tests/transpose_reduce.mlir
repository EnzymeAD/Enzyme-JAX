// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=65536})" %s | FileCheck %s

module {
  func.func private @transpose_reduce(%in: tensor<9x20x45xf64>) -> (tensor<45x20xf64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %reduce = stablehlo.reduce(%in init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<9x20x45xf64>, tensor<f64>) -> tensor<20x45xf64>
    %transpose = "stablehlo.transpose"(%reduce) { permutation = array<i64 : 1, 0> } : (tensor<20x45xf64>) -> tensor<45x20xf64>
    return %transpose : tensor<45x20xf64>
  }
}

// CHECK-LABEL:   func.func private @transpose_reduce(
// CHECK-SAME:                                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<9x20x45xf64>) -> tensor<45x20xf64> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.reduce(%[[VAL_0]] init: %[[VAL_1]]) applies stablehlo.add across dimensions = [0] : (tensor<9x20x45xf64>, tensor<f64>) -> tensor<20x45xf64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.transpose %[[VAL_2]], dims = [1, 0] : (tensor<20x45xf64>) -> tensor<45x20xf64>
// CHECK:           return %[[VAL_3]] : tensor<45x20xf64>
// CHECK:         }

