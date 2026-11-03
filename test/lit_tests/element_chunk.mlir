// RUN: enzymexlamlir-opt --enzyme-hlo-element-chunk="elements=6 chunks=2" %s | FileCheck %s

// Two stages of a contraction over the 4 values of each of 6 elements, read
// from and written to byte buffers: run as a loop over two chunks of 3.
func.func @apply(%x: tensor<192xi8>, %b: tensor<4xf64>) -> tensor<192xi8> {
  %0 = stablehlo.reshape %x : (tensor<192xi8>) -> tensor<24x8xi8>
  %1 = stablehlo.bitcast_convert %0 : (tensor<24x8xi8>) -> tensor<24xf64>
  %2 = stablehlo.reshape %1 : (tensor<24xf64>) -> tensor<6x4xf64>
  %3 = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<4xf64>) -> tensor<6x4xf64>
  %4 = stablehlo.multiply %2, %3 : tensor<6x4xf64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<6x4xf64>, tensor<f64>) -> tensor<6xf64>
  %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<6xf64>) -> tensor<6x4xf64>
  %7 = stablehlo.multiply %6, %3 : tensor<6x4xf64>
  %8 = stablehlo.reshape %7 : (tensor<6x4xf64>) -> tensor<24xf64>
  %9 = stablehlo.bitcast_convert %8 : (tensor<24xf64>) -> tensor<24x8xi8>
  %10 = stablehlo.reshape %9 : (tensor<24x8xi8>) -> tensor<192xi8>
  return %10 : tensor<192xi8>
}

// A reduction across the elements: left alone.
func.func @across_elements(%x: tensor<6x4xf64>) -> tensor<6xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.reduce(%x init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<6x4xf64>, tensor<f64>) -> tensor<4xf64>
  %1 = stablehlo.reshape %0 : (tensor<4xf64>) -> tensor<1x4xf64>
  %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x4xf64>) -> tensor<6x4xf64>
  %3 = stablehlo.multiply %x, %2 : tensor<6x4xf64>
  %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<6x4xf64>, tensor<f64>) -> tensor<6xf64>
  return %4 : tensor<6xf64>
}

// A slice of some of the elements: left alone.
func.func @partial_slice(%x: tensor<6x4xf64>) -> tensor<6x4xf64> {
  %0 = stablehlo.slice %x [1:6, 0:4] : (tensor<6x4xf64>) -> tensor<5x4xf64>
  %1 = stablehlo.slice %x [0:1, 0:4] : (tensor<6x4xf64>) -> tensor<1x4xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<5x4xf64>, tensor<1x4xf64>) -> tensor<6x4xf64>
  return %2 : tensor<6x4xf64>
}

// CHECK-LABEL: func.func @apply(
// CHECK-SAME: %arg0: tensor<192xi8>, %arg1: tensor<4xf64>) -> tensor<192xi8> {
// CHECK-NEXT: %0 = stablehlo.reshape %arg0 : (tensor<192xi8>) -> tensor<24x8xi8>
// CHECK-NEXT: %1 = stablehlo.bitcast_convert %0 : (tensor<24x8xi8>) -> tensor<24xf64>
// CHECK-NEXT: %2 = stablehlo.reshape %1 : (tensor<24xf64>) -> tensor<6x4xf64>
// CHECK-NEXT: %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<6x4xf64>
// CHECK-NEXT: %3:4 = stablehlo.while(%iterArg = %c, %iterArg_1 = %cst_0, %iterArg_2 = %arg1, %iterArg_3 = %2) : tensor<i64>, tensor<6x4xf64>, tensor<4xf64>, tensor<6x4xf64>
// CHECK-NEXT: cond {
// CHECK-NEXT: %c_4 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT: %7 = stablehlo.compare LT, %iterArg, %c_4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT: stablehlo.return %7 : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT: %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %c_5 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT: %7 = stablehlo.multiply %iterArg, %c_5 : tensor<i64>
// CHECK-NEXT: %c_6 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %8 = stablehlo.dynamic_slice %iterArg_3, %7, %c_6, sizes = [3, 4] : (tensor<6x4xf64>, tensor<i64>, tensor<i64>) -> tensor<3x4xf64>
// CHECK-NEXT: %9 = stablehlo.broadcast_in_dim %iterArg_2, dims = [1] : (tensor<4xf64>) -> tensor<3x4xf64>
// CHECK-NEXT: %10 = stablehlo.multiply %8, %9 : tensor<3x4xf64>
// CHECK-NEXT: %11 = stablehlo.reduce(%10 init: %cst_4) applies stablehlo.add across dimensions = [1] : (tensor<3x4xf64>, tensor<f64>) -> tensor<3xf64>
// CHECK-NEXT: %12 = stablehlo.broadcast_in_dim %11, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
// CHECK-NEXT: %13 = stablehlo.multiply %12, %9 : tensor<3x4xf64>
// CHECK-NEXT: %c_7 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %14 = stablehlo.add %iterArg, %c_7 : tensor<i64>
// CHECK-NEXT: %c_8 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT: %15 = stablehlo.multiply %iterArg, %c_8 : tensor<i64>
// CHECK-NEXT: %c_9 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %16 = stablehlo.dynamic_update_slice %iterArg_1, %13, %15, %c_9 : (tensor<6x4xf64>, tensor<3x4xf64>, tensor<i64>, tensor<i64>) -> tensor<6x4xf64>
// CHECK-NEXT: stablehlo.return %14, %16, %iterArg_2, %iterArg_3 : tensor<i64>, tensor<6x4xf64>, tensor<4xf64>, tensor<6x4xf64>
// CHECK-NEXT: }
// CHECK-NEXT: %4 = stablehlo.reshape %3#1 : (tensor<6x4xf64>) -> tensor<24xf64>
// CHECK-NEXT: %5 = stablehlo.bitcast_convert %4 : (tensor<24xf64>) -> tensor<24x8xi8>
// CHECK-NEXT: %6 = stablehlo.reshape %5 : (tensor<24x8xi8>) -> tensor<192xi8>
// CHECK-NEXT: return %6 : tensor<192xi8>
// CHECK-NEXT: }
// CHECK-LABEL: func.func @across_elements(
// CHECK-SAME: %arg0: tensor<6x4xf64>) -> tensor<6xf64> {
// CHECK-NEXT: %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<6x4xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT: %1 = stablehlo.reshape %0 : (tensor<4xf64>) -> tensor<1x4xf64>
// CHECK-NEXT: %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x4xf64>) -> tensor<6x4xf64>
// CHECK-NEXT: %3 = stablehlo.multiply %arg0, %2 : tensor<6x4xf64>
// CHECK-NEXT: %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<6x4xf64>, tensor<f64>) -> tensor<6xf64>
// CHECK-NEXT: return %4 : tensor<6xf64>
// CHECK-NEXT: }
// CHECK-LABEL: func.func @partial_slice(
// CHECK-SAME: %arg0: tensor<6x4xf64>) -> tensor<6x4xf64> {
// CHECK-NEXT: %0 = stablehlo.slice %arg0 [1:6, 0:4] : (tensor<6x4xf64>) -> tensor<5x4xf64>
// CHECK-NEXT: %1 = stablehlo.slice %arg0 [0:1, 0:4] : (tensor<6x4xf64>) -> tensor<1x4xf64>
// CHECK-NEXT: %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<5x4xf64>, tensor<1x4xf64>) -> tensor<6x4xf64>
// CHECK-NEXT: return %2 : tensor<6x4xf64>
// CHECK-NEXT: }
