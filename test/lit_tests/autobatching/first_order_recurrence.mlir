// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK-NOT: stablehlo.while
// CHECK: %[[REDWIN:.*]]:2 = "stablehlo.reduce_window"
// CHECK-SAME: window_dimensions = array<i64: 64>
// CHECK: return %[[REDWIN]]#1 : tensor<64xf64>

func.func @main(%arg0: tensor<64xf64>, %arg1: tensor<64xf64>) -> tensor<64xf64> {
  %c = stablehlo.constant dense<63> : tensor<i64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %c_0 = stablehlo.constant dense<0> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_3 = stablehlo.constant dense<1> : tensor<i32>
  %0 = stablehlo.slice %arg1 [0:1] : (tensor<64xf64>) -> tensor<1xf64>
  %1 = stablehlo.pad %0, %cst, low = [0], high = [63], interior = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<64xf64>
  %2:2 = stablehlo.while(%iterArg = %c_0, %iterArg_4 = %1) : tensor<i64>, tensor<64xf64> attributes {enzyme.disable_mincut}
  cond {
    %3 = stablehlo.compare LT, %iterArg, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %3 : tensor<i1>
  } do {
    %3 = stablehlo.add %c_2, %iterArg {enzymexla.bounds = [[2, 64]]} : tensor<i64>
    %4 = stablehlo.add %iterArg, %c_1 {enzymexla.bounds = [[1, 63]]} : tensor<i64>
    %5 = stablehlo.convert %3 {enzymexla.bounds = [[2, 64]]} : (tensor<i64>) -> tensor<i32>
    %6 = stablehlo.subtract %5, %c_3 {enzymexla.bounds = [[1, 63]]} : tensor<i32>
    %7 = stablehlo.dynamic_slice %arg0, %6, sizes = [1] : (tensor<64xf64>, tensor<i32>) -> tensor<1xf64>
    %8 = stablehlo.subtract %3, %c_1 {enzymexla.bounds = [[1, 63]]} : tensor<i64>
    %9 = stablehlo.convert %8 {enzymexla.bounds = [[1, 63]]} : (tensor<i64>) -> tensor<i32>
    %10 = stablehlo.subtract %9, %c_3 {enzymexla.bounds = [[0, 62]]} : tensor<i32>
    %11 = stablehlo.dynamic_slice %iterArg_4, %10, sizes = [1] : (tensor<64xf64>, tensor<i32>) -> tensor<1xf64>
    %12 = stablehlo.multiply %7, %11 : tensor<1xf64>
    %13 = stablehlo.dynamic_slice %arg1, %6, sizes = [1] : (tensor<64xf64>, tensor<i32>) -> tensor<1xf64>
    %14 = stablehlo.add %12, %13 : tensor<1xf64>
    %15 = stablehlo.dynamic_update_slice %iterArg_4, %14, %6 : (tensor<64xf64>, tensor<1xf64>, tensor<i32>) -> tensor<64xf64>
    stablehlo.return %4, %15 : tensor<i64>, tensor<64xf64>
  }
  return %2#1 : tensor<64xf64>
}
