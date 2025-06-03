// RUN: enzymexlamlir-opt --pass-pipeline='builtin.module(enzyme{postpasses="arith-raise{stablehlo=true},canonicalize,cse,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math,canonicalize,cse,canonicalize"},remove-unnecessary-enzyme-ops,inline,enzyme-hlo-opt)' %s | FileCheck %s

module @reactant_enzyme_... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"Const{typeof(sumabs2first)}(Main.sumabs2first)_autodiff"(%arg0: tensor<5x2x1024xf32>, %arg1: tensor<4x2x1xf32>, %arg2: tensor<4xf32>, %arg3: tensor<16x4x16xcomplex<f32>>) -> (tensor<f32>, tensor<5x2x1024xf32>, tensor<4x2x1xf32>, tensor<4xf32>, tensor<16x4x16xcomplex<f32>>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0]x[o, i, 0]->[0, f, b], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<5x2x1024xf32>, tensor<4x2x1xf32>) -> tensor<1024x4x5xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<4xf32>) -> tensor<1024x4x5xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x4x5xf32>
    %3 = stablehlo.transpose %2, dims = [1, 2, 0] : (tensor<1024x4x5xf32>) -> tensor<4x5x1024xf32>
    %4 = stablehlo.fft %3, type =  RFFT, length = [1024] : (tensor<4x5x1024xf32>) -> tensor<4x5x513xcomplex<f32>>
    %5 = stablehlo.slice %4 [0:4, 0:5, 0:16] : (tensor<4x5x513xcomplex<f32>>) -> tensor<4x5x16xcomplex<f32>>
    %6 = stablehlo.dot_general %arg3, %5, batching_dims = [0] x [2], contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x4x16xcomplex<f32>>, tensor<4x5x16xcomplex<f32>>) -> tensor<16x16x5xcomplex<f32>>
    %7 = stablehlo.transpose %6, dims = [1, 2, 0] : (tensor<16x16x5xcomplex<f32>>) -> tensor<16x5x16xcomplex<f32>>
    %8 = stablehlo.pad %7, %cst_0, low = [0, 0, 0], high = [0, 0, 497], interior = [0, 0, 0] : (tensor<16x5x16xcomplex<f32>>, tensor<complex<f32>>) -> tensor<16x5x513xcomplex<f32>>
    %9 = stablehlo.fft %8, type =  IRFFT, length = [1024] : (tensor<16x5x513xcomplex<f32>>) -> tensor<16x5x1024xf32>
    %10 = stablehlo.multiply %9, %9 : tensor<16x5x1024xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<16x5x1024xf32>, tensor<f32>) -> tensor<f32>
    return %11, %arg0, %arg1, %arg2, %arg3 : tensor<f32>, tensor<5x2x1024xf32>, tensor<4x2x1xf32>, tensor<4xf32>, tensor<16x4x16xcomplex<f32>>
  }
  func.func @main(%arg0: tensor<5x2x1024xf32> {tf.aliasing_output = 4 : i32}, %arg1: tensor<4x2x1xf32> {tf.aliasing_output = 5 : i32}, %arg2: tensor<4xf32> {tf.aliasing_output = 6 : i32}, %arg3: tensor<16x4x16xcomplex<f32>> {tf.aliasing_output = 7 : i32}) -> (tensor<5x2x1024xf32>, tensor<4x2x1xf32>, tensor<4xf32>, tensor<16x4x16xcomplex<f32>>, tensor<5x2x1024xf32>, tensor<4x2x1xf32>, tensor<4xf32>, tensor<16x4x16xcomplex<f32>>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<5x2x1024xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<4x2x1xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
    %cst_3 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<16x4x16xcomplex<f32>>
    %0:8 = enzyme.autodiff @"Const{typeof(sumabs2first)}(Main.sumabs2first)_autodiff"(%arg0, %arg1, %arg2, %arg3, %cst, %cst_0, %cst_1, %cst_2, %cst_3) {activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>]} : (tensor<5x2x1024xf32>, tensor<4x2x1xf32>, tensor<4xf32>, tensor<16x4x16xcomplex<f32>>, tensor<f32>, tensor<5x2x1024xf32>, tensor<4x2x1xf32>, tensor<4xf32>, tensor<16x4x16xcomplex<f32>>) -> (tensor<5x2x1024xf32>, tensor<4x2x1xf32>, tensor<4xf32>, tensor<16x4x16xcomplex<f32>>, tensor<5x2x1024xf32>, tensor<4x2x1xf32>, tensor<4xf32>, tensor<16x4x16xcomplex<f32>>)
    return %0#4, %0#5, %0#6, %0#7, %0#0, %0#1, %0#2, %0#3 : tensor<5x2x1024xf32>, tensor<4x2x1xf32>, tensor<4xf32>, tensor<16x4x16xcomplex<f32>>, tensor<5x2x1024xf32>, tensor<4x2x1xf32>, tensor<4xf32>, tensor<16x4x16xcomplex<f32>>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x2x1024xf32> {tf.aliasing_output = 4 : i32}, %arg1: tensor<4x2x1xf32> {tf.aliasing_output = 5 : i32}, %arg2: tensor<4xf32> {tf.aliasing_output = 6 : i32}, %arg3: tensor<16x4x16xcomplex<f32>> {tf.aliasing_output = 7 : i32}) -> (tensor<5x2x1024xf32>, tensor<4x2x1xf32>, tensor<4xf32>, tensor<16x4x16xcomplex<f32>>, tensor<5x2x1024xf32>, tensor<4x2x1xf32>, tensor<4xf32>, tensor<16x4x16xcomplex<f32>>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<[9.765625E-4, 0.001953125, 0.001953125, 0.001953125, 0.001953125, 0.001953125, 0.001953125, 0.001953125, 0.001953125, 0.001953125, 0.001953125, 0.001953125, 0.001953125, 0.001953125, 0.001953125, 0.001953125]> : tensor<16xf32>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
// CHECK-NEXT:     %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT{LITERAL}:     %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0]x[o, i, 0]->[0, f, b], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<5x2x1024xf32>, tensor<4x2x1xf32>) -> tensor<1024x4x5xf32>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<4xf32>) -> tensor<1024x4x5xf32>
// CHECK-NEXT:     %2 = stablehlo.add %0, %1 : tensor<1024x4x5xf32>
// CHECK-NEXT:     %3 = stablehlo.transpose %2, dims = [1, 2, 0] : (tensor<1024x4x5xf32>) -> tensor<4x5x1024xf32>
// CHECK-NEXT:     %4 = stablehlo.fft %3, type =  RFFT, length = [1024] : (tensor<4x5x1024xf32>) -> tensor<4x5x513xcomplex<f32>>
// CHECK-NEXT:     %5 = stablehlo.slice %4 [0:4, 0:5, 0:16] : (tensor<4x5x513xcomplex<f32>>) -> tensor<4x5x16xcomplex<f32>>
// CHECK-NEXT:     %6 = stablehlo.dot_general %arg3, %5, batching_dims = [0] x [2], contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x4x16xcomplex<f32>>, tensor<4x5x16xcomplex<f32>>) -> tensor<16x16x5xcomplex<f32>>
// CHECK-NEXT:     %7 = stablehlo.transpose %6, dims = [1, 2, 0] : (tensor<16x16x5xcomplex<f32>>) -> tensor<16x5x16xcomplex<f32>>
// CHECK-NEXT:     %8 = stablehlo.pad %7, %cst_1, low = [0, 0, 0], high = [0, 0, 497], interior = [0, 0, 0] : (tensor<16x5x16xcomplex<f32>>, tensor<complex<f32>>) -> tensor<16x5x513xcomplex<f32>>
// CHECK-NEXT:     %9 = stablehlo.fft %8, type =  IRFFT, length = [1024] : (tensor<16x5x513xcomplex<f32>>) -> tensor<16x5x1024xf32>
// CHECK-NEXT:     %10 = stablehlo.add %9, %9 : tensor<16x5x1024xf32>
// CHECK-NEXT:     %11 = stablehlo.fft %10, type =  RFFT, length = [1024] : (tensor<16x5x1024xf32>) -> tensor<16x5x513xcomplex<f32>>
// CHECK-NEXT:     %12 = stablehlo.slice %11 [0:16, 0:5, 0:16] : (tensor<16x5x513xcomplex<f32>>) -> tensor<16x5x16xcomplex<f32>>
// CHECK-NEXT:     %13 = stablehlo.complex %cst_0, %cst : tensor<16xcomplex<f32>>
// CHECK-NEXT:     %14 = stablehlo.broadcast_in_dim %13, dims = [2] : (tensor<16xcomplex<f32>>) -> tensor<16x5x16xcomplex<f32>>
// CHECK-NEXT:     %15 = stablehlo.multiply %12, %14 : tensor<16x5x16xcomplex<f32>>
// CHECK-NEXT:     %16 = chlo.conj %15 : tensor<16x5x16xcomplex<f32>> -> tensor<16x5x16xcomplex<f32>>
// CHECK-NEXT:     %17 = stablehlo.dot_general %16, %5, batching_dims = [2] x [2], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x5x16xcomplex<f32>>, tensor<4x5x16xcomplex<f32>>) -> tensor<16x16x4xcomplex<f32>>
// CHECK-NEXT:     %18 = chlo.conj %17 : tensor<16x16x4xcomplex<f32>> -> tensor<16x16x4xcomplex<f32>>
// CHECK-NEXT:     %19 = stablehlo.transpose %18, dims = [0, 2, 1] : (tensor<16x16x4xcomplex<f32>>) -> tensor<16x4x16xcomplex<f32>>
// CHECK-NEXT:     %20 = stablehlo.dot_general %16, %arg3, batching_dims = [2] x [0], contracting_dims = [0] x [2], precision = [DEFAULT, DEFAULT] : (tensor<16x5x16xcomplex<f32>>, tensor<16x4x16xcomplex<f32>>) -> tensor<16x5x4xcomplex<f32>>
// CHECK-NEXT:     %21 = chlo.conj %20 : tensor<16x5x4xcomplex<f32>> -> tensor<16x5x4xcomplex<f32>>
// CHECK-NEXT:     %22 = stablehlo.transpose %21, dims = [2, 1, 0] : (tensor<16x5x4xcomplex<f32>>) -> tensor<4x5x16xcomplex<f32>>
// CHECK-NEXT:     %23 = stablehlo.pad %22, %cst_1, low = [0, 0, 0], high = [0, 0, 497], interior = [0, 0, 0] : (tensor<4x5x16xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4x5x513xcomplex<f32>>
// CHECK-NEXT:     %24 = chlo.conj %23 : tensor<4x5x513xcomplex<f32>> -> tensor<4x5x513xcomplex<f32>>
// CHECK-NEXT:     %25 = stablehlo.pad %24, %cst_1, low = [0, 0, 0], high = [0, 0, 511], interior = [0, 0, 0] : (tensor<4x5x513xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4x5x1024xcomplex<f32>>
// CHECK-NEXT:     %26 = stablehlo.fft %25, type =  FFT, length = [1024] : (tensor<4x5x1024xcomplex<f32>>) -> tensor<4x5x1024xcomplex<f32>>
// CHECK-NEXT:     %27 = stablehlo.real %26 : (tensor<4x5x1024xcomplex<f32>>) -> tensor<4x5x1024xf32>
// CHECK-NEXT:     %28 = stablehlo.reduce(%27 init: %cst_2) applies stablehlo.add across dimensions = [2, 1] : (tensor<4x5x1024xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT{LITERAL}:     %29 = stablehlo.convolution(%27, %arg1) dim_numbers = [f, b, 0]x[i, o, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], lhs_dilate = [1], rhs_dilate = [1], reverse = [true]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<4x5x1024xf32>, tensor<4x2x1xf32>) -> tensor<5x2x1024xf32>
// CHECK-NEXT{LITERAL}:     %30 = stablehlo.convolution(%arg0, %27) dim_numbers = [f, b, 0]x[o, i, 0]->[f, b, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<5x2x1024xf32>, tensor<4x5x1024xf32>) -> tensor<4x2x1xf32>
// CHECK-NEXT:     return %29, %30, %28, %19, %arg0, %arg1, %arg2, %arg3 : tensor<5x2x1024xf32>, tensor<4x2x1xf32>, tensor<4xf32>, tensor<16x4x16xcomplex<f32>>, tensor<5x2x1024xf32>, tensor<4x2x1xf32>, tensor<4xf32>, tensor<16x4x16xcomplex<f32>>
// CHECK-NEXT:   }
