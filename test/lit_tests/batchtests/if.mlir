// RUN: enzymexlamlir-opt %s --enzyme-batch --arith-raise --canonicalize | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-batch --arith-raise | stablehlo-translate - --interpret

module {
  func.func private @relu_broadcast_scalar(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %pred = stablehlo.compare GE, %arg0, %cst : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %result = "stablehlo.if"(%pred) ({
        stablehlo.return %arg0 : tensor<f64>
    }, {
        %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        stablehlo.return %cst_0 : tensor<f64>
    }) :  (tensor<i1>) -> tensor<f64>
    return %result : tensor<f64>
  }
  func.func @main() {
    %arg0 = stablehlo.constant dense<[[42.0, -42.0], [0.0, 1.0]]> : tensor<2x2xf64>
    %0 = enzyme.batch @relu_broadcast_scalar(%arg0) {batch_shape = array<i64: 2, 2>} : (tensor<2x2xf64>) -> tensor<2x2xf64>
    check.expect_eq_const %0, dense<[[42.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf64>
    return
  }
}

// CHECK:  func.func private @batched_relu_broadcast_scalar(%arg0: tensor<2x2xf64>) -> tensor<2x2xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf64>
// CHECK-NEXT:    %0 = stablehlo.compare  GE, %arg0, %cst_3 : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xi1>
// CHECK-NEXT:    %1:2 = stablehlo.while(%iterArg = %c_2, %iterArg_4 = %cst_3) : tensor<i64>, tensor<2x2xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %2 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      %3 = stablehlo.divide %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      %4 = stablehlo.remainder %3, %c : tensor<i64>
// CHECK-NEXT:      %5 = stablehlo.divide %iterArg, %c : tensor<i64>
// CHECK-NEXT:      %6 = stablehlo.remainder %5, %c : tensor<i64>
// CHECK-NEXT:      %7 = stablehlo.dynamic_slice %0, %4, %6, sizes = [1, 1] : (tensor<2x2xi1>, tensor<i64>, tensor<i64>) -> tensor<1x1xi1>
// CHECK-NEXT:      %8 = stablehlo.reshape %7 : (tensor<1x1xi1>) -> tensor<i1>
// CHECK-NEXT:      %9 = stablehlo.dynamic_slice %arg0, %4, %6, sizes = [1, 1] : (tensor<2x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
// CHECK-NEXT:      %10 = stablehlo.reshape %9 : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:      %11 = "stablehlo.if"(%8) ({
// CHECK-NEXT:        stablehlo.return %10 : tensor<f64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %cst : tensor<f64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<f64>
// CHECK-NEXT:      %12 = stablehlo.reshape %11 : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:      %13 = stablehlo.dynamic_update_slice %iterArg_4, %12, %4, %6 : (tensor<2x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<2x2xf64>
// CHECK-NEXT:      stablehlo.return %2, %13 : tensor<i64>, tensor<2x2xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %1#1 : tensor<2x2xf64>
// CHECK-NEXT:  }
