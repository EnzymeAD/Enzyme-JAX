// RUN: enzymexlamlir-opt %s --enzyme-batch --arith-raise="stablehlo=true" --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-batch --arith-raise="stablehlo=true" --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func private @reduce_bcast_init(%arg0: tensor<3xf64>) -> tensor<f64> {
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
  %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<f64>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
  return %1 : tensor<f64>
}

func.func private @reduce_varying_init(%arg0: tensor<3xf64>, %init: tensor<f64>) -> tensor<f64> {
  %0 = stablehlo.reduce(%arg0 init: %init) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
  return %0 : tensor<f64>
}

func.func @main() {
  %in = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %res0 = enzyme.batch @reduce_bcast_init(%in) <{batch_shape = array<i64: 2>}> : (tensor<2x3xf64>) -> tensor<2xf64>
  %exp0 = stablehlo.constant dense<[7.0, 16.0]> : tensor<2xf64>
  "check.expect_almost_eq"(%res0, %exp0) : (tensor<2xf64>, tensor<2xf64>) -> ()

  %inits = stablehlo.constant dense<[10.0, 100.0]> : tensor<2xf64>
  %res1 = enzyme.batch @reduce_varying_init(%in, %inits) <{batch_shape = array<i64: 2>}> : (tensor<2x3xf64>, tensor<2xf64>) -> tensor<2xf64>
  %exp1 = stablehlo.constant dense<[16.0, 115.0]> : tensor<2xf64>
  "check.expect_almost_eq"(%res1, %exp1) : (tensor<2xf64>, tensor<2xf64>) -> ()
  return
}

// CHECK-LABEL: func.func private @batched_reduce_bcast_init(
// CHECK-SAME: %arg0: tensor<2x3xf64>) -> tensor<2xf64>
// CHECK: %[[CST:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK: %[[RED:.+]] = stablehlo.reduce(%arg0 init: %[[CST]]) applies stablehlo.add across dimensions = [1] : (tensor<2x3xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK: return %[[RED]] : tensor<2xf64>
