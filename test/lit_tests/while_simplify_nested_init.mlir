// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// A raised dot product: the inner loop starts its vector variables from the
// outer loop's, which start from the layout ops on the arguments. Neither
// loop updates them, so both arguments come back as themselves.
func.func @main(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>, %arg2: tensor<8xi8>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> (tensor<32xi8>, tensor<32xi8>, tensor<8xi8>) {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_0 = stablehlo.constant dense<1> : tensor<i64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.reshape %arg0 : (tensor<32xi8>) -> tensor<4x8xi8>
  %1 = stablehlo.bitcast_convert %0 : (tensor<4x8xi8>) -> tensor<4xf64>
  %2 = stablehlo.reshape %arg1 : (tensor<32xi8>) -> tensor<4x8xi8>
  %3 = stablehlo.bitcast_convert %2 : (tensor<4x8xi8>) -> tensor<4xf64>
  %4 = stablehlo.reshape %arg2 : (tensor<8xi8>) -> tensor<1x8xi8>
  %5 = stablehlo.bitcast_convert %4 : (tensor<1x8xi8>) -> tensor<1xf64>
  %6 = stablehlo.convert %arg3 : (tensor<i32>) -> tensor<i64>
  %7 = stablehlo.convert %arg4 : (tensor<i32>) -> tensor<i64>
  %8:4 = stablehlo.while(%iterArg = %c, %iterArg_1 = %1, %iterArg_2 = %3, %iterArg_3 = %5) : tensor<i64>, tensor<4xf64>, tensor<4xf64>, tensor<1xf64>
  cond {
    %13 = stablehlo.compare LT, %iterArg, %6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %13 : tensor<i1>
  } do {
    %13:4 = stablehlo.while(%iterArg_4 = %c, %iterArg_5 = %cst, %iterArg_6 = %iterArg_1, %iterArg_7 = %iterArg_2) : tensor<i64>, tensor<f64>, tensor<4xf64>, tensor<4xf64>
    cond {
      %16 = stablehlo.compare LT, %iterArg_4, %7 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %16 : tensor<i1>
    } do {
      %16 = stablehlo.dynamic_slice %iterArg_6, %iterArg_4, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
      %17 = stablehlo.dynamic_slice %iterArg_7, %iterArg_4, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
      %18 = stablehlo.multiply %16, %17 : tensor<1xf64>
      %19 = stablehlo.reshape %18 : (tensor<1xf64>) -> tensor<f64>
      %20 = stablehlo.add %iterArg_5, %19 : tensor<f64>
      %21 = stablehlo.add %iterArg_4, %c_0 : tensor<i64>
      stablehlo.return %21, %20, %iterArg_6, %iterArg_7 : tensor<i64>, tensor<f64>, tensor<4xf64>, tensor<4xf64>
    }
    %14 = stablehlo.reshape %13#1 : (tensor<f64>) -> tensor<1xf64>
    %15 = stablehlo.add %iterArg, %c_0 : tensor<i64>
    stablehlo.return %15, %13#2, %13#3, %14 : tensor<i64>, tensor<4xf64>, tensor<4xf64>, tensor<1xf64>
  }
  %9 = stablehlo.bitcast_convert %8#1 : (tensor<4xf64>) -> tensor<4x8xi8>
  %10 = stablehlo.reshape %9 : (tensor<4x8xi8>) -> tensor<32xi8>
  %11 = stablehlo.bitcast_convert %8#2 : (tensor<4xf64>) -> tensor<4x8xi8>
  %12 = stablehlo.reshape %11 : (tensor<4x8xi8>) -> tensor<32xi8>
  %13 = stablehlo.bitcast_convert %8#3 : (tensor<1xf64>) -> tensor<1x8xi8>
  %14 = stablehlo.reshape %13 : (tensor<1x8xi8>) -> tensor<8xi8>
  return %10, %12, %14 : tensor<32xi8>, tensor<32xi8>, tensor<8xi8>
}

// CHECK: return %arg0, %arg1, %{{.+}} : tensor<32xi8>, tensor<32xi8>, tensor<8xi8>
