// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s 

module {
  func.func private @"Const{typeof(simple_reduce)}(simple_reduce)_autodiff"(%arg0: tensor<5x3xf32>) -> tensor<3xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %a1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<5x3xf32>, tensor<f32>) -> tensor<3xf32>
    return %a1 : tensor<3xf32>
  }
  func.func @main(%arg0: tensor<5x3xf32>) -> (tensor<5x3xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
    %0 = enzyme.autodiff @"Const{typeof(simple_reduce)}(simple_reduce)_autodiff"(%arg0, %cst) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<5x3xf32>, tensor<3xf32>) -> (tensor<5x3xf32>)
    return %0 : tensor<5x3xf32>
  }
}

// CHECK:  func.func private @"diffeConst{typeof(simple_reduce)}(simple_reduce)_autodiff"(%arg0: tensor<5x3xf32>, %arg1: tensor<3xf32>) -> tensor<5x3xf32> {
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<3xf32>
// CHECK-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<5x3xf32>
// CHECK-NEXT:    %0 = arith.addf %arg1, %cst : tensor<3xf32>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<3xf32>) -> tensor<5x3xf32>
// CHECK-NEXT:    %2 = arith.addf %1, %cst_0 : tensor<5x3xf32>
// CHECK-NEXT:    return %2 : tensor<5x3xf32>
// CHECK-NEXT:  }
