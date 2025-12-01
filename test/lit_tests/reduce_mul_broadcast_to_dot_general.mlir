// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<100x100xf64>, %arg1: tensor<100x100xf64>) -> tensor<100x100xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<100x100xf64>) -> tensor<100x100x100xf64>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1, 0] : (tensor<100x100xf64>) -> tensor<100x100x100xf64>
    %2 = stablehlo.multiply %0, %1 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<100x100x100xf64>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<100x100x100xf64>, tensor<f64>) -> tensor<100x100xf64>
    return %3 : tensor<100x100xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<100x100xf64>, %arg1: tensor<100x100xf64>) -> tensor<100x100xf64> {
// CHECK-NEXT:    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [1] : (tensor<100x100xf64>, tensor<100x100xf64>) -> tensor<100x100xf64>
// CHECK-NEXT:    return %0 : tensor<100x100xf64>
// CHECK-NEXT:  }
