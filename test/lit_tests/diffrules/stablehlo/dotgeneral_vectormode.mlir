// RUN: enzymexlamlir-opt --enzyme --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func private @neuralnetwork(%arg0: tensor<2x4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4x3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<4x2xf32>) -> (tensor<4x3xf32>, tensor<2x4xf32>, tensor<4xf32>, tensor<4x3xf32>, tensor<3xf32>, tensor<4x2xf32>) {
    %0 = stablehlo.dot_general %arg0, %arg4, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<4x4xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<4xf32>) -> tensor<4x4xf32>
    %2 = stablehlo.add %0, %1 : tensor<4x4xf32>
    %3 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<3xf32>) -> tensor<3x4xf32>
    %4 = stablehlo.dot_general %arg2, %2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x3xf32>, tensor<4x4xf32>) -> tensor<3x4xf32>
    %5 = stablehlo.add %4, %3 : tensor<3x4xf32>
    %6 = stablehlo.transpose %5, dims = [1, 0] : (tensor<3x4xf32>) -> tensor<4x3xf32>
    return %6, %arg0, %arg1, %arg2, %arg3, %arg4 : tensor<4x3xf32>, tensor<2x4xf32>, tensor<4xf32>, tensor<4x3xf32>, tensor<3xf32>, tensor<4x2xf32>
  }
  func.func @main(%arg0: tensor<2x4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4x3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<4x2xf32>) -> (tensor<4x2x3xf32>, tensor<2x4xf32>, tensor<4xf32>, tensor<4x3xf32>, tensor<3xf32>) {
    %cst = stablehlo.constant dense<[[[1.000000e+00, 0.000000e+00], [1.000000e+00, 0.000000e+00], [1.000000e+00, 0.000000e+00], [1.000000e+00, 0.000000e+00]], [[0.000000e+00, 1.000000e+00], [0.000000e+00, 1.000000e+00], [0.000000e+00, 1.000000e+00], [0.000000e+00, 1.000000e+00]]]> : tensor<2x4x2xf32>
    %0:5 = enzyme.fwddiff @neuralnetwork(%arg0, %arg1, %arg2, %arg3, %arg4, %cst) {activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_constnoneed>], width = 2 : i64} : (tensor<2x4xf32>, tensor<4xf32>, tensor<4x3xf32>, tensor<3xf32>, tensor<4x2xf32>, tensor<2x4x2xf32>) -> (tensor<2x4x3xf32>, tensor<2x4xf32>, tensor<4xf32>, tensor<4x3xf32>, tensor<3xf32>)
    %1 = stablehlo.slice %0#0 [1:2, 0:4, 0:3] : (tensor<2x4x3xf32>) -> tensor<1x4x3xf32>
    %2 = stablehlo.transpose %1, dims = [2, 1, 0] : (tensor<1x4x3xf32>) -> tensor<3x4x1xf32>
    %3 = stablehlo.slice %0#0 [0:1, 0:4, 0:3] : (tensor<2x4x3xf32>) -> tensor<1x4x3xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] : (tensor<1x4x3xf32>) -> tensor<3x4x1xf32>
    %5 = stablehlo.reshape %4 : (tensor<3x4x1xf32>) -> tensor<3x1x4xf32>
    %6 = stablehlo.reshape %2 : (tensor<3x4x1xf32>) -> tensor<3x1x4xf32>
    %7 = stablehlo.concatenate %5, %6, dim = 1 : (tensor<3x1x4xf32>, tensor<3x1x4xf32>) -> tensor<3x2x4xf32>
    %8 = stablehlo.transpose %7, dims = [2, 1, 0] : (tensor<3x2x4xf32>) -> tensor<4x2x3xf32>
    return %8, %0#1, %0#2, %0#3, %0#4 : tensor<4x2x3xf32>, tensor<2x4xf32>, tensor<4xf32>, tensor<4x3xf32>, tensor<3xf32>
  }
}

// CHECK: func.func private @fwddiffe2neuralnetwork(%arg0: tensor<2x4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4x3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<4x2xf32>, %arg5: tensor<2x4x2xf32>) -> (tensor<2x4x3xf32>, tensor<2x4xf32>, tensor<4xf32>, tensor<4x3xf32>, tensor<3xf32>) {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<3x4xf32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
// CHECK-NEXT:   %0 = "enzyme.broadcast"(%cst_0) <{shape = array<i64: 2>}> : (tensor<4x4xf32>) -> tensor<2x4x4xf32>
// CHECK-NEXT:   %1 = "enzyme.broadcast"(%arg0) <{shape = array<i64: 2>}> : (tensor<2x4xf32>) -> tensor<2x2x4xf32>
// CHECK-NEXT:   %2 = stablehlo.dot_general %1, %arg5, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x4xf32>, tensor<2x4x2xf32>) -> tensor<2x4x4xf32>
// CHECK-NEXT:   %3 = stablehlo.add %0, %2 : tensor<2x4x4xf32>
// CHECK-NEXT:   %4 = "enzyme.broadcast"(%cst) <{shape = array<i64: 2>}> : (tensor<3x4xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:   %5 = "enzyme.broadcast"(%arg2) <{shape = array<i64: 2>}> : (tensor<4x3xf32>) -> tensor<2x4x3xf32>
// CHECK-NEXT:   %6 = stablehlo.dot_general %5, %3, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x3xf32>, tensor<2x4x4xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:   %7 = stablehlo.add %4, %6 : tensor<2x3x4xf32>
// CHECK-NEXT:   %8 = stablehlo.transpose %7, dims = [0, 2, 1] : (tensor<2x3x4xf32>) -> tensor<2x4x3xf32>
// CHECK-NEXT:   return %8, %arg0, %arg1, %arg2, %arg3 : tensor<2x4x3xf32>, tensor<2x4xf32>, tensor<4xf32>, tensor<4x3xf32>, tensor<3xf32>
// CHECK-NEXT: }

module {
  func.func private @"Const{typeof(fn)}(Main.fn)_autodiff"(%arg0: tensor<2x3xf32>) -> (tensor<f32>, tensor<2x3xf32>) {
    %0 = stablehlo.dot_general %arg0, %arg0, contracting_dims = [0, 1] x [0, 1] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<f32>
    return %0, %arg0 : tensor<f32>, tensor<2x3xf32>
  }
  func.func @main(%arg0: tensor<2x3xf32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<2x3xf32>) {
    %cst = stablehlo.constant dense<[[[1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 1.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e+00]]]> : tensor<6x2x3xf32>
    %0:2 = enzyme.fwddiff @"Const{typeof(fn)}(Main.fn)_autodiff"(%arg0, %cst) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>, #enzyme<activity enzyme_const>], width = 6 : i64} : (tensor<2x3xf32>, tensor<6x2x3xf32>) -> (tensor<6xf32>, tensor<2x3xf32>)
    %1 = stablehlo.slice %0#0 [0:1] : (tensor<6xf32>) -> tensor<1xf32>
    %2 = stablehlo.reshape %1 : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.slice %0#0 [1:2] : (tensor<6xf32>) -> tensor<1xf32>
    %4 = stablehlo.reshape %3 : (tensor<1xf32>) -> tensor<f32>
    %5 = stablehlo.slice %0#0 [2:3] : (tensor<6xf32>) -> tensor<1xf32>
    %6 = stablehlo.reshape %5 : (tensor<1xf32>) -> tensor<f32>
    %7 = stablehlo.slice %0#0 [3:4] : (tensor<6xf32>) -> tensor<1xf32>
    %8 = stablehlo.reshape %7 : (tensor<1xf32>) -> tensor<f32>
    %9 = stablehlo.slice %0#0 [4:5] : (tensor<6xf32>) -> tensor<1xf32>
    %10 = stablehlo.reshape %9 : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.slice %0#0 [5:6] : (tensor<6xf32>) -> tensor<1xf32>
    %12 = stablehlo.reshape %11 : (tensor<1xf32>) -> tensor<f32>
    return %2, %4, %6, %8, %10, %12, %0#1 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<2x3xf32>
  }
}

// CHECK: func.func private @"fwddiffe6Const{typeof(fn)}(Main.fn)_autodiff"(%arg0: tensor<2x3xf32>, %arg1: tensor<6x2x3xf32>) -> (tensor<6xf32>, tensor<2x3xf32>) {
// CHECK-NEXT:   %0 = "enzyme.broadcast"(%arg0) <{shape = array<i64: 6>}> : (tensor<2x3xf32>) -> tensor<6x2x3xf32>
// CHECK-NEXT:   %1 = stablehlo.dot_general %arg1, %0, batching_dims = [0] x [0], contracting_dims = [1, 2] x [1, 2] : (tensor<6x2x3xf32>, tensor<6x2x3xf32>) -> tensor<6xf32>
// CHECK-NEXT:   %2 = "enzyme.broadcast"(%arg0) <{shape = array<i64: 6>}> : (tensor<2x3xf32>) -> tensor<6x2x3xf32>
// CHECK-NEXT:   %3 = stablehlo.dot_general %2, %arg1, batching_dims = [0] x [0], contracting_dims = [1, 2] x [1, 2] : (tensor<6x2x3xf32>, tensor<6x2x3xf32>) -> tensor<6xf32>
// CHECK-NEXT:   %4 = stablehlo.add %1, %3 : tensor<6xf32>
// CHECK-NEXT:   return %4, %arg0 : tensor<6xf32>, tensor<2x3xf32>
// CHECK-NEXT: }
