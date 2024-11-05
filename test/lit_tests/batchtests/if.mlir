// RUN: enzymexlamlir-opt %s --enzyme-batch --arith-raise --canonicalize | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-batch --arith-raise | %stablehlo-translate - --interpret

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
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf64>
// CHECK-NEXT:    %0 = stablehlo.compare  GE, %arg0, %cst_1 : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xi1>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:1, 0:1] : (tensor<2x2xi1>) -> tensor<1x1xi1>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<1x1xi1>) -> tensor<i1>
// CHECK-NEXT:    %3 = stablehlo.slice %arg0 [0:1, 0:1] : (tensor<2x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %4 = stablehlo.reshape %3 : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %5 = "stablehlo.if"(%2) ({
// CHECK-NEXT:      stablehlo.return %4 : tensor<f64>
// CHECK-NEXT:    }, {
// CHECK-NEXT:      stablehlo.return %cst : tensor<f64>
// CHECK-NEXT:    }) : (tensor<i1>) -> tensor<f64>
// CHECK-NEXT:    %6 = stablehlo.reshape %5 : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %7 = stablehlo.dynamic_update_slice %cst_1, %6, %c_0, %c_0 : (tensor<2x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %8 = stablehlo.slice %0 [1:2, 0:1] : (tensor<2x2xi1>) -> tensor<1x1xi1>
// CHECK-NEXT:    %9 = stablehlo.reshape %8 : (tensor<1x1xi1>) -> tensor<i1>
// CHECK-NEXT:    %10 = stablehlo.slice %arg0 [1:2, 0:1] : (tensor<2x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %11 = stablehlo.reshape %10 : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %12 = "stablehlo.if"(%9) ({
// CHECK-NEXT:      stablehlo.return %11 : tensor<f64>
// CHECK-NEXT:    }, {
// CHECK-NEXT:      stablehlo.return %cst : tensor<f64>
// CHECK-NEXT:    }) : (tensor<i1>) -> tensor<f64>
// CHECK-NEXT:    %13 = stablehlo.reshape %12 : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %14 = stablehlo.dynamic_update_slice %7, %13, %c, %c_0 : (tensor<2x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %15 = stablehlo.slice %0 [0:1, 1:2] : (tensor<2x2xi1>) -> tensor<1x1xi1>
// CHECK-NEXT:    %16 = stablehlo.reshape %15 : (tensor<1x1xi1>) -> tensor<i1>
// CHECK-NEXT:    %17 = stablehlo.slice %arg0 [0:1, 1:2] : (tensor<2x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %18 = stablehlo.reshape %17 : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %19 = "stablehlo.if"(%16) ({
// CHECK-NEXT:      stablehlo.return %18 : tensor<f64>
// CHECK-NEXT:    }, {
// CHECK-NEXT:      stablehlo.return %cst : tensor<f64>
// CHECK-NEXT:    }) : (tensor<i1>) -> tensor<f64>
// CHECK-NEXT:    %20 = stablehlo.reshape %19 : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %21 = stablehlo.dynamic_update_slice %14, %20, %c_0, %c : (tensor<2x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %22 = stablehlo.slice %0 [1:2, 1:2] : (tensor<2x2xi1>) -> tensor<1x1xi1>
// CHECK-NEXT:    %23 = stablehlo.reshape %22 : (tensor<1x1xi1>) -> tensor<i1>
// CHECK-NEXT:    %24 = stablehlo.slice %arg0 [1:2, 1:2] : (tensor<2x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %25 = stablehlo.reshape %24 : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %26 = "stablehlo.if"(%23) ({
// CHECK-NEXT:      stablehlo.return %25 : tensor<f64>
// CHECK-NEXT:    }, {
// CHECK-NEXT:      stablehlo.return %cst : tensor<f64>
// CHECK-NEXT:    }) : (tensor<i1>) -> tensor<f64>
// CHECK-NEXT:    %27 = stablehlo.reshape %26 : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %28 = stablehlo.dynamic_update_slice %21, %27, %c, %c : (tensor<2x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<2x2xf64>
// CHECK-NEXT:    return %28 : tensor<2x2xf64>
// CHECK-NEXT:  }
