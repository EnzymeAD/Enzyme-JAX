// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --enzyme-hlo-opt | FileCheck %s

module {
  func.func private @"Const{var\22#35#39\22{typeof(loss)}}(var\22#35#39\22{typeof(loss)}(Main.loss, Core.Box(ConvTranspose((3, 3), 3 => 2, stride=2))))_autodiff"(%arg0: tensor<3x2x3x3xf32>, %arg1: tensor<2xf32>, %arg2: tensor<1x3x5x5xf32>) -> (tensor<f32>, tensor<3x2x3x3xf32>, tensor<2xf32>, tensor<1x3x5x5xf32>) {
    %cst = stablehlo.constant dense<2.420000e+02> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convolution(%arg2, %arg0) dim_numbers = [b, f, 1, 0]x[i, o, 1, 0]->[0, 1, f, b], window = {pad = [[2, 2], [2, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x5x5xf32>, tensor<3x2x3x3xf32>) -> tensor<11x11x2x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<2xf32>) -> tensor<11x11x2x1xf32>
    %2 = stablehlo.add %0, %1 : tensor<11x11x2x1xf32>
    %3 = stablehlo.reduce(%2 init: %cst_0) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<11x11x2x1xf32>, tensor<f32>) -> tensor<f32>
    %4 = stablehlo.divide %3, %cst : tensor<f32>
    return %4, %arg0, %arg1, %arg2 : tensor<f32>, tensor<3x2x3x3xf32>, tensor<2xf32>, tensor<1x3x5x5xf32>
  }
  func.func @enzyme_withgradient(%arg0: tensor<3x2x3x3xf32>, %arg1: tensor<2xf32>, %arg2: tensor<1x3x5x5xf32>) -> (tensor<f32>, tensor<1x3x5x5xf32>, tensor<3x2x3x3xf32>, tensor<2xf32>, tensor<1x3x5x5xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1x3x5x5xf32>
    %0:5 = enzyme.autodiff @"Const{var\22#35#39\22{typeof(loss)}}(var\22#35#39\22{typeof(loss)}(Main.loss, Core.Box(ConvTranspose((3, 3), 3 => 2, stride=2))))_autodiff"(%arg0, %arg1, %arg2, %cst, %cst_0) {activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>]} : (tensor<3x2x3x3xf32>, tensor<2xf32>, tensor<1x3x5x5xf32>, tensor<f32>, tensor<1x3x5x5xf32>) -> (tensor<f32>, tensor<3x2x3x3xf32>, tensor<2xf32>, tensor<1x3x5x5xf32>, tensor<1x3x5x5xf32>)
    return %0#0, %0#4, %0#1, %0#2, %0#3 : tensor<f32>, tensor<1x3x5x5xf32>, tensor<3x2x3x3xf32>, tensor<2xf32>, tensor<1x3x5x5xf32>
  }
}

// CHECK:  func.func private @"diffeConst{var\22#35#39\22{typeof(loss)}}(var\22#35#39\22{typeof(loss)}(Main.loss, Core.Box(ConvTranspose((3, 3), 3 => 2, stride=2))))_autodiff"(%arg0: tensor<3x2x3x3xf32>, %arg1: tensor<2xf32>, %arg2: tensor<1x3x5x5xf32>, %arg3: tensor<f32>, %arg4: tensor<1x3x5x5xf32>) -> (tensor<f32>, tensor<3x2x3x3xf32>, tensor<2xf32>, tensor<1x3x5x5xf32>, tensor<1x3x5x5xf32>) {
// CHECK-NEXT:    %[[zero:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[cst242:.+]] = stablehlo.constant dense<2.420000e+02> : tensor<f32>
// CHECK-NEXT:    %[[v0:.+]] = stablehlo.convolution(%arg2, %arg0) dim_numbers = [b, f, 1, 0]x[i, o, 1, 0]->[0, 1, f, b], window = {pad = {{\[\[}}2, 2], [2, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x5x5xf32>, tensor<3x2x3x3xf32>) -> tensor<11x11x2x1xf32>
// CHECK-NEXT:    %[[v1:.+]] = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<2xf32>) -> tensor<11x11x2x1xf32>
// CHECK-NEXT:    %[[v2:.+]] = stablehlo.add %[[v0]], %[[v1]] : tensor<11x11x2x1xf32>
// CHECK-NEXT:    %[[v3:.+]] = stablehlo.reduce(%[[v2]] init: %[[zero]]) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<11x11x2x1xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    %[[v4:.+]] = stablehlo.divide %[[v3]], %[[cst242]] : tensor<f32>
// CHECK-NEXT:    %[[v5:.+]] = stablehlo.divide %arg3, %[[cst242]] : tensor<f32>
// CHECK-NEXT:    %[[v6:.+]] = stablehlo.broadcast_in_dim %[[v5]], dims = [] : (tensor<f32>) -> tensor<11x11x2x1xf32>
// CHECK-NEXT:    %[[v7:.+]] = stablehlo.convolution(%[[v6]], %arg0) dim_numbers = [0, 1, f, b]x[o, i, 1, 0]->[b, f, 1, 0], window = {stride = [2, 2], pad = {{\[\[}}0, 0], [0, 0]], rhs_dilate = [1, 1], reverse = [true, true]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<11x11x2x1xf32>, tensor<3x2x3x3xf32>) -> tensor<1x3x5x5xf32>
// CHECK-NEXT:    %[[v8:.+]] = arith.addf %arg4, %[[v7]] : tensor<1x3x5x5xf32>
// CHECK-NEXT:    return %[[v4]], %arg0, %arg1, %arg2, %[[v8]] : tensor<f32>, tensor<3x2x3x3xf32>, tensor<2xf32>, tensor<1x3x5x5xf32>, tensor<1x3x5x5xf32>
// CHECK-NEXT:  }
