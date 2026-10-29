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
    %3:3 = enzyme.fwddiff @fwd_autodiff(%arg0, %2) <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>, #enzyme.activity<enzyme_dup>], width = 2 : i64}> : (tensor<2xf64>, tensor<2x2xf64>) -> (tensor<2xf64>, tensor<2xf64>, tensor<2x2xf64>)
    %4 = stablehlo.slice %3#0 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
    %5 = stablehlo.slice %3#0 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
    return %4, %5, %3#1, %arg1, %arg2 : tensor<1xf64>, tensor<1xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>
}

// CHECK:  func.func @main(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>, %arg2: tensor<2xf64>) -> (tensor<1xf64>, tensor<1xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg1, %arg2, dim = 0 : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<4xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %3:3 = call @fwddiffe2fwd_autodiff(%arg0, %2) : (tensor<2xf64>, tensor<2x2xf64>) -> (tensor<2xf64>, tensor<2xf64>, tensor<2x2xf64>)
// CHECK-NEXT:    %4 = stablehlo.slice %3#0 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %5 = stablehlo.slice %3#0 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT:    return %4, %5, %3#1, %arg1, %arg2 : tensor<1xf64>, tensor<1xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>
// CHECK-NEXT:  }
// CHECK:  func.func private @fwddiffe2fwd_autodiff(%arg0: tensor<2xf64>, %arg1: tensor<2x2xf64>) -> (tensor<2xf64>, tensor<2xf64>, tensor<2x2xf64>) {
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = "enzyme.broadcast"(%arg0) <{shape = array<i64: 2>}> : (tensor<2xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %1 = stablehlo.multiply %arg1, %0 : tensor<2x2xf64>
// CHECK-NEXT:    %2 = "enzyme.broadcast"(%arg0) <{shape = array<i64: 2>}> : (tensor<2xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %3 = stablehlo.multiply %arg1, %2 : tensor<2x2xf64>
// CHECK-NEXT:    %4 = arith.addf %1, %3 fastmath<fast> : tensor<2x2xf64>
// CHECK-NEXT:    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:    return %5, %arg0, %arg1 : tensor<2xf64>, tensor<2xf64>, tensor<2x2xf64>
// CHECK-NEXT:  }

func.func private @fwd_reduce_window(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[3, 0]]> : tensor<1x2xi64>, window_dimensions = array<i64: 4>}> ({
    ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f64>
      stablehlo.return %1 : tensor<f64>
    }) : (tensor<4xf64>, tensor<f64>) -> tensor<4xf64>
    return %0 : tensor<4xf64>
}
func.func @batch_reduce_window(%arg0: tensor<4xf64>, %arg1: tensor<2x4xf64>) -> (tensor<4xf64>, tensor<2x4xf64>) {
    %0:2 = enzyme.fwddiff @fwd_reduce_window(%arg0, %arg1) <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dup>], width = 2 : i64}> : (tensor<4xf64>, tensor<2x4xf64>) -> (tensor<4xf64>, tensor<2x4xf64>)
    return %0#0, %0#1 : tensor<4xf64>, tensor<2x4xf64>
}

// CHECK:  func.func private @fwddiffe2fwd_reduce_window(%arg0: tensor<4xf64>, %arg1: tensor<2x4xf64>) -> (tensor<4xf64>, tensor<2x4xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = "stablehlo.reduce_window"(%arg1, %cst_0) <{padding = dense<{{\[\[}}0, 0], [3, 0{{\]\]}}> : tensor<2x2xi64>, window_dimensions = array<i64: 1, 4>}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      %2 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %2 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<2x4xf64>, tensor<f64>) -> tensor<2x4xf64>
// CHECK-NEXT:    %1 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<{{\[\[}}3, 0{{\]\]}}> : tensor<1x2xi64>, window_dimensions = array<i64: 4>}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      %2 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %2 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<4xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:    return %1, %0 : tensor<4xf64>, tensor<2x4xf64>
// CHECK-NEXT:  }
