// RUN: enzymexlamlir-opt --pass-pipeline='builtin.module(enzyme{postpasses="arith-raise{stablehlo=true},canonicalize,cse,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math,canonicalize,cse,canonicalize"},remove-unnecessary-enzyme-ops,inline,enzyme-hlo-opt)' %s | FileCheck %s

module @"reactant_\E2\88\87test_fft" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"Const{typeof(test_fft)}(Main.test_fft)_autodiff"(%arg0: tensor<3x4xf32>) -> (tensor<f32>, tensor<3x4xf32>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.fft %arg0, type =  RFFT, length = [3, 4] : (tensor<3x4xf32>) -> tensor<3x3xcomplex<f32>>
    %1 = stablehlo.fft %0, type =  IRFFT, length = [3, 4] : (tensor<3x3xcomplex<f32>>) -> tensor<3x4xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<3x4xf32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<3x4xf32>, tensor<f32>) -> tensor<f32>
    return %3, %arg0 : tensor<f32>, tensor<3x4xf32>
  }
  func.func @main(%arg0: tensor<3x4xf32> {tf.aliasing_output = 1 : i32}) -> (tensor<3x4xf32>, tensor<3x4xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<3x4xf32>
    %0:2 = enzyme.autodiff @"Const{typeof(test_fft)}(Main.test_fft)_autodiff"(%arg0, %cst, %cst_0) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_active>]} : (tensor<3x4xf32>, tensor<f32>, tensor<3x4xf32>) -> (tensor<3x4xf32>, tensor<3x4xf32>)
    return %0#1, %0#0 : tensor<3x4xf32>, tensor<3x4xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3x4xf32> {tf.aliasing_output = 1 : i32}) -> (tensor<3x4xf32>, tensor<3x4xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<[0.0833333358, 0.166666672, 0.0833333358]> : tensor<3xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>

// CHECK-NEXT:     %0 = stablehlo.fft %arg0, type =  RFFT, length = [3, 4] : (tensor<3x4xf32>) -> tensor<3x3xcomplex<f32>>
// CHECK-NEXT:     %1 = stablehlo.fft %0, type =  IRFFT, length = [3, 4] : (tensor<3x3xcomplex<f32>>) -> tensor<3x4xf32>
// CHECK-NEXT:     %2 = stablehlo.add %1, %1 : tensor<3x4xf32>

// CHECK-NEXT:     %3 = stablehlo.fft %2, type =  RFFT, length = [3, 4] : (tensor<3x4xf32>) -> tensor<3x3xcomplex<f32>>
// CHECK-NEXT:     %4 = stablehlo.complex %cst, %cst_1 : tensor<3xcomplex<f32>>
// CHECK-NEXT:     %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<3xcomplex<f32>>) -> tensor<3x3xcomplex<f32>>
// CHECK-NEXT:     %6 = stablehlo.multiply %3, %5 : tensor<3x3xcomplex<f32>>

// CHECK-NEXT:     %7 = chlo.conj %6 : tensor<3x3xcomplex<f32>> -> tensor<3x3xcomplex<f32>>
// CHECK-NEXT:     %8 = stablehlo.pad %7, %cst_0, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<3x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3x4xcomplex<f32>>
// CHECK-NEXT:     %9 = stablehlo.fft %8, type =  FFT, length = [3, 4] : (tensor<3x4xcomplex<f32>>) -> tensor<3x4xcomplex<f32>>
// CHECK-NEXT:     %10 = stablehlo.real %9 : (tensor<3x4xcomplex<f32>>) -> tensor<3x4xf32>

// CHECK-NEXT:     return %10, %arg0 : tensor<3x4xf32>, tensor<3x4xf32>
// CHECK-NEXT: }
