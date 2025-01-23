// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt)" %s | FileCheck %s 

func.func private @identity_broadcast_scalar(%arg0: tensor<f8E4M3FN>) -> tensor<f8E4M3FN> {
    return %arg0 : tensor<f8E4M3FN>
}

func.func @main(%arg0: tensor<3x10xf8E4M3FN>) -> (tensor<f8E4M3FN>, tensor<3x10xf8E4M3FN>) {
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x10xf8E4M3FN>) -> tensor<10x3xf8E4M3FN>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
  %1 = stablehlo.convert %cst : (tensor<f16>) -> tensor<f8E4M3FN>
  %2 = enzyme.batch @identity_broadcast_scalar(%0) {batch_shape = array<i64: 10, 3>} : (tensor<10x3xf8E4M3FN>) -> tensor<10x3xf8E4M3FN>
  %3 = stablehlo.convert %2 : tensor<10x3xf8E4M3FN>
  %4 = stablehlo.reduce(%3 init: %1) applies stablehlo.add across dimensions = [0, 1] : (tensor<10x3xf8E4M3FN>, tensor<f8E4M3FN>) -> tensor<f8E4M3FN>
  %5 = stablehlo.transpose %0, dims = [1, 0] : (tensor<10x3xf8E4M3FN>) -> tensor<3x10xf8E4M3FN>
  return %4, %5 : tensor<f8E4M3FN>, tensor<3x10xf8E4M3FN>
}

// CHECK-LABEL: main
// CHECK: %{{.+}} = stablehlo.constant dense<0.000000e+00> : tensor<f8E4M3FN>

