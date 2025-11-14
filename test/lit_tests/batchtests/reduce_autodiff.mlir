// RUN: enzymexlamlir-opt %s --enzyme --enzyme-hlo-opt | FileCheck %s

func.func private @fwd_autodiff(%arg0: tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<2xf64>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
    return %1, %arg0 : tensor<f64>, tensor<2xf64>
}
func.func @main(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>, %arg2: tensor<2xf64>) -> (tensor<1xf64>, tensor<1xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) {
    %0 = stablehlo.concatenate %arg1, %arg2, dim = 0 : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
    %1 = stablehlo.reshape %0 : (tensor<4xf64>) -> tensor<2x2xf64>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
    %3:3 = enzyme.fwddiff @fwd_autodiff(%arg0, %2) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>, #enzyme<activity enzyme_dup>], width = 2 : i64} : (tensor<2xf64>, tensor<2x2xf64>) -> (tensor<2xf64>, tensor<2xf64>, tensor<2x2xf64>)
    %4 = stablehlo.slice %3#0 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
    %5 = stablehlo.slice %3#0 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
    return %4, %5, %3#1, %arg1, %arg2 : tensor<1xf64>, tensor<1xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>
}

// CHECK:  func.func @main(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>, %arg2: tensor<2xf64>) -> (tensor<1xf64>, tensor<1xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg1, %arg2, dim = 0 : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 {enzymexla.guaranteed_symmetric = false} : (tensor<4xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %3:3 = call @fwddiffe2fwd_autodiff(%arg0, %2) : (tensor<2xf64>, tensor<2x2xf64>) -> (tensor<2xf64>, tensor<2xf64>, tensor<2x2xf64>)
// CHECK-NEXT:    %4 = stablehlo.slice %3#0 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %5 = stablehlo.slice %3#0 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT:    return %4, %5, %3#1, %arg1, %arg2 : tensor<1xf64>, tensor<1xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @fwddiffe2fwd_autodiff(%arg0: tensor<2xf64>, %arg1: tensor<2x2xf64>) -> (tensor<2xf64>, tensor<2xf64>, tensor<2x2xf64>) {
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = "enzyme.broadcast"(%arg0) <{shape = array<i64: 2>}> : (tensor<2xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %1 = stablehlo.multiply %arg1, %0 : tensor<2x2xf64>
// CHECK-NEXT:    %2 = "enzyme.broadcast"(%arg0) <{shape = array<i64: 2>}> : (tensor<2xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %3 = stablehlo.multiply %arg1, %2 : tensor<2x2xf64>
// CHECK-NEXT:    %4 = arith.addf %1, %3 : tensor<2x2xf64>
// CHECK-NEXT:    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:    return %5, %arg0, %arg1 : tensor<2xf64>, tensor<2xf64>, tensor<2x2xf64>
// CHECK-NEXT:  }
