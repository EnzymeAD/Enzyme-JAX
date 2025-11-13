// RUN: enzymexla-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<32x32xf64>, %arg1: tensor<32x32xf64>) -> tensor<f64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1] : (tensor<32x32xf64>, tensor<32x32xf64>) -> tensor<32x32xf64>
  %1 = stablehlo.reshape %0 : (tensor<32x32xf64>) -> tensor<1024xf64>
  %2 = stablehlo.slice %1 [0:1024:33] : (tensor<1024xf64>) -> tensor<32xf64>
  %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32xf64>, tensor<f64>) -> tensor<f64>
  return %3 : tensor<f64>
}

func.func @main2(%arg0: tensor<4x32xf64>, %arg1: tensor<32x4xf64>) -> tensor<f64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf64>, tensor<4x32xf64>) -> tensor<32x32xf64>
  %1 = stablehlo.reshape %0 : (tensor<32x32xf64>) -> tensor<1024xf64>
  %2 = stablehlo.slice %1 [0:1024:33] : (tensor<1024xf64>) -> tensor<32xf64>
  %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32xf64>, tensor<f64>) -> tensor<f64>
  return %3 : tensor<f64>
}

func.func @main3(%arg0: tensor<4x32xf64>, %arg1: tensor<7x4xf64>) -> tensor<5xf64> {
  %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<7x4xf64>, tensor<4x32xf64>) -> tensor<7x32xf64>
  %1 = stablehlo.reshape %0 : (tensor<7x32xf64>) -> tensor<224xf64>
  %2 = stablehlo.slice %1 [66:224:33] : (tensor<224xf64>) -> tensor<5xf64>
  %3 = stablehlo.slice %1 [0:134:33] : (tensor<224xf64>) -> tensor<5xf64>
  %4 = stablehlo.add %2, %3 : tensor<5xf64>
  return %4 : tensor<5xf64>
}

func.func @fail1(%arg0: tensor<4x32xf64>, %arg1: tensor<7x4xf64>) -> tensor<5xf64> {
  %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<7x4xf64>, tensor<4x32xf64>) -> tensor<7x32xf64>
  %1 = stablehlo.reshape %0 : (tensor<7x32xf64>) -> tensor<224xf64>
  %2 = stablehlo.slice %1 [67:224:33] : (tensor<224xf64>) -> tensor<5xf64>
  return %2 : tensor<5xf64>
}

// TODO: test for strided diagonal access
