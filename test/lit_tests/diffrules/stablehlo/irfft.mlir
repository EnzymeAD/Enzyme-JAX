// RUN: enzymexlamlir-opt --pass-pipeline='builtin.module(enzyme{postpasses="arith-raise{stablehlo=true},canonicalize,cse,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math,canonicalize,cse,canonicalize"},remove-unnecessary-enzyme-ops,inline,enzyme-hlo-opt)' %s | FileCheck %s

module @reactant_gradient attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"Const{typeof(test_fn3)}(Main.test_fn3)_autodiff"(%arg0: tensor<5x2x4xcomplex<f32>>) -> (tensor<f32>, tensor<5x2x4xcomplex<f32>>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<5x2x4xcomplex<f32>>) -> tensor<4x5x2xcomplex<f32>>
    %1 = stablehlo.fft %0, type =  IRFFT, length = [5, 3] : (tensor<4x5x2xcomplex<f32>>) -> tensor<4x5x3xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<4x5x3xf32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<4x5x3xf32>, tensor<f32>) -> tensor<f32>
    return %3, %arg0 : tensor<f32>, tensor<5x2x4xcomplex<f32>>
  }
  func.func @main(%arg0: tensor<5x2x4xcomplex<f32>> {tf.aliasing_output = 1 : i32}) -> (tensor<5x2x4xcomplex<f32>>, tensor<5x2x4xcomplex<f32>>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<5x2x4xcomplex<f32>>
    %0:2 = enzyme.autodiff @"Const{typeof(test_fn3)}(Main.test_fn3)_autodiff"(%arg0, %cst, %cst_0) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_active>]} : (tensor<5x2x4xcomplex<f32>>, tensor<f32>, tensor<5x2x4xcomplex<f32>>) -> (tensor<5x2x4xcomplex<f32>>, tensor<5x2x4xcomplex<f32>>)
    return %0#1, %0#0 : tensor<5x2x4xcomplex<f32>>, tensor<5x2x4xcomplex<f32>>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x2x4xcomplex<f32>> {tf.aliasing_output = 1 : i32}) -> (tensor<5x2x4xcomplex<f32>>, tensor<5x2x4xcomplex<f32>>) {
// CHECK-NEXT{LITERAL}:     %cst = stablehlo.constant dense<[[[(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)], [(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)], [(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)], [(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)], [(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)]], [[(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)], [(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)], [(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)], [(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)], [(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)]], [[(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)], [(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)], [(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)], [(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)], [(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)]], [[(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)], [(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)], [(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)], [(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)], [(0.0666666701,0.000000e+00), (0.13333334,0.000000e+00)]]]> : tensor<4x5x2xcomplex<f32>>
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<5x2x4xcomplex<f32>>) -> tensor<4x5x2xcomplex<f32>>
// CHECK-NEXT:     %1 = stablehlo.fft %0, type =  IRFFT, length = [5, 3] : (tensor<4x5x2xcomplex<f32>>) -> tensor<4x5x3xf32>
// CHECK-NEXT:     %2 = stablehlo.add %1, %1 : tensor<4x5x3xf32>
// CHECK-NEXT:     %3 = stablehlo.fft %2, type =  RFFT, length = [5, 3] : (tensor<4x5x3xf32>) -> tensor<4x5x2xcomplex<f32>>
// CHECK-NEXT:     %4 = stablehlo.multiply %3, %cst : tensor<4x5x2xcomplex<f32>>
// CHECK-NEXT:     %5 = stablehlo.transpose %4, dims = [1, 2, 0] : (tensor<4x5x2xcomplex<f32>>) -> tensor<5x2x4xcomplex<f32>>
// CHECK-NEXT:     return %5, %arg0 : tensor<5x2x4xcomplex<f32>>, tensor<5x2x4xcomplex<f32>>
// CHECK-NEXT:   }
