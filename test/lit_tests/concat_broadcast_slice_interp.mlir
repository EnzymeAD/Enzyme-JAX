// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s
// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | stablehlo-translate - --interpret

func.func @slice_bcast_concat_interplay(%x: tensor<4xf64>) -> tensor<4xf64> {
  %cst = stablehlo.constant dense<2.0> : tensor<4xf64>
  %s0 = stablehlo.slice %x [2:3] : (tensor<4xf64>) -> tensor<1xf64>
  %s1 = stablehlo.slice %x [0:1] : (tensor<4xf64>) -> tensor<1xf64>
  %b0 = stablehlo.broadcast_in_dim %s0, dims = [0] : (tensor<1xf64>) -> tensor<4xf64>
  %b1 = stablehlo.broadcast_in_dim %s1, dims = [0] : (tensor<1xf64>) -> tensor<4xf64>
  %c0 = stablehlo.slice %b1 [0:2] : (tensor<4xf64>) -> tensor<2xf64>
  %c1 = stablehlo.slice %b0 [2:4] : (tensor<4xf64>) -> tensor<2xf64>
  %cat = stablehlo.concatenate %c0, %c1, dim = 0 : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
  %mul = stablehlo.multiply %cst, %cat : tensor<4xf64>
  return %mul : tensor<4xf64>
}

// CHECK-LABEL: func.func @slice_bcast_concat_interplay

func.func @main() {
  %x = stablehlo.constant dense<[3.0, 99.0, 7.0, -5.0]> : tensor<4xf64>
  %got = func.call @slice_bcast_concat_interplay(%x) : (tensor<4xf64>) -> tensor<4xf64>
  check.expect_almost_eq_const %got, dense<[6.0, 6.0, 14.0, 14.0]> : tensor<4xf64>

  %y = stablehlo.constant dense<[[1.0], [2.0], [3.0], [4.0]]> : tensor<4x1xf64>
  %s0 = stablehlo.slice %y [0:2, 0:1] : (tensor<4x1xf64>) -> tensor<2x1xf64>
  %b0 = stablehlo.broadcast_in_dim %s0, dims = [0, 1] : (tensor<2x1xf64>) -> tensor<2x3xf64>
  %s1 = stablehlo.slice %y [2:4, 0:1] : (tensor<4x1xf64>) -> tensor<2x1xf64>
  %b1 = stablehlo.broadcast_in_dim %s1, dims = [0, 1] : (tensor<2x1xf64>) -> tensor<2x3xf64>
  %cat2 = stablehlo.concatenate %b0, %b1, dim = 0 : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<4x3xf64>
  check.expect_almost_eq_const %cat2, dense<[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]> : tensor<4x3xf64>
  return
}
