// RUN: enzymexlamlir-opt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --inline --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func private @"Const{typeof(slicing)}(Main.slicing)_autodiff"(%arg0: tensor<1x4x1xf32>) -> (tensor<f32>, tensor<1x4x1xf32>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
    %0 = stablehlo.slice %arg0 [0:1, 0:1, 0:1] : (tensor<1x4x1xf32>) -> tensor<1x1x1xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<1xf32>) -> tensor<3xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<3xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<3xf32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    return %5, %arg0 : tensor<f32>, tensor<1x4x1xf32>
  }
  func.func @main(%arg0: tensor<1x4x1xf32>) -> (tensor<1x4x1xf32>, tensor<1x4x1xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0:2 = enzyme.autodiff @"Const{typeof(slicing)}(Main.slicing)_autodiff"(%arg0, %cst) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_const>]} : (tensor<1x4x1xf32>, tensor<f32>) -> (tensor<1x4x1xf32>, tensor<1x4x1xf32>)
    return %0#1, %0#0 : tensor<1x4x1xf32>, tensor<1x4x1xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<1x4x1xf32>) -> (tensor<1x4x1xf32>, tensor<1x4x1xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:1, 0:1, 0:1] : (tensor<1x4x1xf32>) -> tensor<1x1x1xf32>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<1x1x1xf32>) -> tensor<1xf32>
// CHECK-NEXT:     %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<1xf32>) -> tensor<3xf32>
// CHECK-NEXT:     %3 = stablehlo.add %2, %2 : tensor<3xf32>
// CHECK-NEXT:     %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:     %5 = stablehlo.reshape %4 : (tensor<f32>) -> tensor<1x1x1xf32>
// CHECK-NEXT:     %6 = stablehlo.pad %5, %cst, low = [0, 0, 0], high = [0, 3, 0], interior = [0, 0, 0] : (tensor<1x1x1xf32>, tensor<f32>) -> tensor<1x4x1xf32>
// CHECK-NEXT:     return %6, %arg0 : tensor<1x4x1xf32>, tensor<1x4x1xf32>
// CHECK-NEXT:   }
