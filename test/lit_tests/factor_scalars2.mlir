// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=reshape_empty_broadcast<1>;transpose_broadcast_in_dim_to_broadcast_in_dim<16>;factor_scalars_in_dot_general;reshape_elementwise(1);transpose_elementwise(1)},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<f64>, %arg1: tensor<1x1024x64xbf16>, %arg2: tensor<4x1024x512xbf16>, %arg3: tensor<1x1024x64xbf16>, %arg4: tensor<4x1024x2048xbf16>) -> tensor<4x32x1024x1024xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<true> : tensor<1024x1024xi1>
    %0 = stablehlo.iota dim = 0 : tensor<1024xi64>
    %1 = stablehlo.iota dim = 0 : tensor<1024xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %2 = stablehlo.broadcast %c_1, sizes = [1024] : (tensor<i64>) -> tensor<1024xi64>
    %3 = stablehlo.add %1, %2 : tensor<1024xi64>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1024xi64>) -> tensor<1024x1024xi64>
    %5 = stablehlo.broadcast %0, sizes = [1024] : (tensor<1024xi64>) -> tensor<1024x1024xi64>
    %6 = stablehlo.compare  GE, %4, %5 : (tensor<1024x1024xi64>, tensor<1024x1024xi64>) -> tensor<1024x1024xi1>
    %c_2 = stablehlo.constant dense<false> : tensor<1024x1024xi1>
    %7 = stablehlo.select %6, %c, %c_2 : tensor<1024x1024xi1>, tensor<1024x1024xi1>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x1024xf32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x1024xf32>
    %10 = stablehlo.select %7, %8, %9 : tensor<1024x1024xi1>, tensor<1024x1024xf32>
    %11 = stablehlo.reshape %arg1 : (tensor<1x1024x64xbf16>) -> tensor<1024x64x1x1xbf16>
    %12 = stablehlo.transpose %11, dims = [2, 3, 0, 1] : (tensor<1024x64x1x1xbf16>) -> tensor<1x1x1024x64xbf16>
    %13 = stablehlo.reshape %arg2 : (tensor<4x1024x512xbf16>) -> tensor<4x1024x8x64xbf16>
    %14 = stablehlo.slice %13 [0:4, 0:1024, 0:8, 0:32] : (tensor<4x1024x8x64xbf16>) -> tensor<4x1024x8x32xbf16>
    %15 = stablehlo.transpose %14, dims = [0, 2, 1, 3] : (tensor<4x1024x8x32xbf16>) -> tensor<4x8x1024x32xbf16>
    %16 = stablehlo.reshape %arg2 : (tensor<4x1024x512xbf16>) -> tensor<4x1024x8x64xbf16>
    %17 = stablehlo.slice %16 [0:4, 0:1024, 0:8, 32:64] : (tensor<4x1024x8x64xbf16>) -> tensor<4x1024x8x32xbf16>
    %18 = stablehlo.transpose %17, dims = [0, 2, 1, 3] : (tensor<4x1024x8x32xbf16>) -> tensor<4x8x1024x32xbf16>
    %19 = stablehlo.negate %18 : tensor<4x8x1024x32xbf16>
    %20 = stablehlo.transpose %19, dims = [0, 2, 1, 3] : (tensor<4x8x1024x32xbf16>) -> tensor<4x1024x8x32xbf16>
    %21 = stablehlo.reshape %20 : (tensor<4x1024x8x32xbf16>) -> tensor<1048576xbf16>
    %22 = stablehlo.reshape %21 : (tensor<1048576xbf16>) -> tensor<4x1024x8x32xbf16>
    %23 = stablehlo.transpose %22, dims = [0, 2, 1, 3] : (tensor<4x1024x8x32xbf16>) -> tensor<4x8x1024x32xbf16>
    %24 = stablehlo.concatenate %23, %15, dim = 3 : (tensor<4x8x1024x32xbf16>, tensor<4x8x1024x32xbf16>) -> tensor<4x8x1024x64xbf16>
    %25 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2, 3] : (tensor<1x1x1024x64xbf16>) -> tensor<4x8x1024x64xbf16>
    %26 = stablehlo.multiply %24, %25 : tensor<4x8x1024x64xbf16>
    %27 = stablehlo.reshape %arg3 : (tensor<1x1024x64xbf16>) -> tensor<1024x64x1x1xbf16>
    %28 = stablehlo.transpose %27, dims = [2, 3, 0, 1] : (tensor<1024x64x1x1xbf16>) -> tensor<1x1x1024x64xbf16>
    %29 = stablehlo.reshape %arg2 : (tensor<4x1024x512xbf16>) -> tensor<4x1024x8x64xbf16>
    %30 = stablehlo.transpose %29, dims = [0, 2, 1, 3] : (tensor<4x1024x8x64xbf16>) -> tensor<4x8x1024x64xbf16>
    %31 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2, 3] : (tensor<1x1x1024x64xbf16>) -> tensor<4x8x1024x64xbf16>
    %32 = stablehlo.multiply %30, %31 : tensor<4x8x1024x64xbf16>
    %33 = stablehlo.transpose %32, dims = [0, 2, 1, 3] : (tensor<4x8x1024x64xbf16>) -> tensor<4x1024x8x64xbf16>
    %34 = stablehlo.reshape %33 : (tensor<4x1024x8x64xbf16>) -> tensor<2097152xbf16>
    %35 = stablehlo.reshape %34 : (tensor<2097152xbf16>) -> tensor<4x1024x8x64xbf16>
    %36 = stablehlo.transpose %35, dims = [0, 2, 1, 3] : (tensor<4x1024x8x64xbf16>) -> tensor<4x8x1024x64xbf16>
    %37 = stablehlo.add %36, %26 : tensor<4x8x1024x64xbf16>
    %38 = stablehlo.transpose %37, dims = [0, 2, 1, 3] : (tensor<4x8x1024x64xbf16>) -> tensor<4x1024x8x64xbf16>
    %39 = stablehlo.reshape %38 : (tensor<4x1024x8x64xbf16>) -> tensor<2097152xbf16>
    %40 = stablehlo.reshape %39 : (tensor<2097152xbf16>) -> tensor<4x1024x8x64xbf16>
    %41 = stablehlo.transpose %40, dims = [0, 2, 1, 3] : (tensor<4x1024x8x64xbf16>) -> tensor<4x8x1024x64xbf16>
    %42 = stablehlo.concatenate %41, dim = 2 : (tensor<4x8x1024x64xbf16>) -> tensor<4x8x1024x64xbf16>
    %43 = stablehlo.convert %42 : (tensor<4x8x1024x64xbf16>) -> tensor<4x8x1024x64xf32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 3, 4] : (tensor<4x8x1024x64xf32>) -> tensor<4x8x4x1024x64xf32>
    %45 = stablehlo.reshape %44 : (tensor<4x8x4x1024x64xf32>) -> tensor<4x32x1024x64xf32>
    %46 = stablehlo.transpose %45, dims = [0, 1, 3, 2] : (tensor<4x32x1024x64xf32>) -> tensor<4x32x64x1024xf32>
    %47 = stablehlo.convert %arg0 : (tensor<f64>) -> tensor<f32>
    %48 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<f32>) -> tensor<4x32x64x1024xf32>
    %49 = stablehlo.multiply %46, %48 : tensor<4x32x64x1024xf32>
    %50 = stablehlo.transpose %49, dims = [0, 1, 3, 2] : (tensor<4x32x64x1024xf32>) -> tensor<4x32x1024x64xf32>
    %51 = stablehlo.reshape %50 : (tensor<4x32x1024x64xf32>) -> tensor<8388608xf32>
    %52 = stablehlo.reshape %51 : (tensor<8388608xf32>) -> tensor<4x32x1024x64xf32>
    %53 = stablehlo.transpose %52, dims = [0, 1, 3, 2] : (tensor<4x32x1024x64xf32>) -> tensor<4x32x64x1024xf32>
    %54 = stablehlo.reshape %arg1 : (tensor<1x1024x64xbf16>) -> tensor<1024x64x1x1xbf16>
    %55 = stablehlo.transpose %54, dims = [2, 3, 0, 1] : (tensor<1024x64x1x1xbf16>) -> tensor<1x1x1024x64xbf16>
    %56 = stablehlo.reshape %arg4 : (tensor<4x1024x2048xbf16>) -> tensor<4x1024x32x64xbf16>
    %57 = stablehlo.slice %56 [0:4, 0:1024, 0:32, 0:32] : (tensor<4x1024x32x64xbf16>) -> tensor<4x1024x32x32xbf16>
    %58 = stablehlo.transpose %57, dims = [0, 2, 1, 3] : (tensor<4x1024x32x32xbf16>) -> tensor<4x32x1024x32xbf16>
    %59 = stablehlo.reshape %arg4 : (tensor<4x1024x2048xbf16>) -> tensor<4x1024x32x64xbf16>
    %60 = stablehlo.slice %59 [0:4, 0:1024, 0:32, 32:64] : (tensor<4x1024x32x64xbf16>) -> tensor<4x1024x32x32xbf16>
    %61 = stablehlo.transpose %60, dims = [0, 2, 1, 3] : (tensor<4x1024x32x32xbf16>) -> tensor<4x32x1024x32xbf16>
    %62 = stablehlo.negate %61 : tensor<4x32x1024x32xbf16>
    %63 = stablehlo.transpose %62, dims = [0, 2, 1, 3] : (tensor<4x32x1024x32xbf16>) -> tensor<4x1024x32x32xbf16>
    %64 = stablehlo.reshape %63 : (tensor<4x1024x32x32xbf16>) -> tensor<4194304xbf16>
    %65 = stablehlo.reshape %64 : (tensor<4194304xbf16>) -> tensor<4x1024x32x32xbf16>
    %66 = stablehlo.transpose %65, dims = [0, 2, 1, 3] : (tensor<4x1024x32x32xbf16>) -> tensor<4x32x1024x32xbf16>
    %67 = stablehlo.concatenate %66, %58, dim = 3 : (tensor<4x32x1024x32xbf16>, tensor<4x32x1024x32xbf16>) -> tensor<4x32x1024x64xbf16>
    %68 = stablehlo.broadcast_in_dim %55, dims = [0, 1, 2, 3] : (tensor<1x1x1024x64xbf16>) -> tensor<4x32x1024x64xbf16>
    %69 = stablehlo.multiply %67, %68 : tensor<4x32x1024x64xbf16>
    %70 = stablehlo.reshape %arg3 : (tensor<1x1024x64xbf16>) -> tensor<1024x64x1x1xbf16>
    %71 = stablehlo.transpose %70, dims = [2, 3, 0, 1] : (tensor<1024x64x1x1xbf16>) -> tensor<1x1x1024x64xbf16>
    %72 = stablehlo.reshape %arg4 : (tensor<4x1024x2048xbf16>) -> tensor<4x1024x32x64xbf16>
    %73 = stablehlo.transpose %72, dims = [0, 2, 1, 3] : (tensor<4x1024x32x64xbf16>) -> tensor<4x32x1024x64xbf16>
    %74 = stablehlo.broadcast_in_dim %71, dims = [0, 1, 2, 3] : (tensor<1x1x1024x64xbf16>) -> tensor<4x32x1024x64xbf16>
    %75 = stablehlo.multiply %73, %74 : tensor<4x32x1024x64xbf16>
    %76 = stablehlo.transpose %75, dims = [0, 2, 1, 3] : (tensor<4x32x1024x64xbf16>) -> tensor<4x1024x32x64xbf16>
    %77 = stablehlo.reshape %76 : (tensor<4x1024x32x64xbf16>) -> tensor<8388608xbf16>
    %78 = stablehlo.reshape %77 : (tensor<8388608xbf16>) -> tensor<4x1024x32x64xbf16>
    %79 = stablehlo.transpose %78, dims = [0, 2, 1, 3] : (tensor<4x1024x32x64xbf16>) -> tensor<4x32x1024x64xbf16>
    %80 = stablehlo.add %79, %69 : tensor<4x32x1024x64xbf16>
    %81 = stablehlo.transpose %80, dims = [0, 2, 1, 3] : (tensor<4x32x1024x64xbf16>) -> tensor<4x1024x32x64xbf16>
    %82 = stablehlo.reshape %81 : (tensor<4x1024x32x64xbf16>) -> tensor<8388608xbf16>
    %83 = stablehlo.reshape %82 : (tensor<8388608xbf16>) -> tensor<4x1024x32x64xbf16>
    %84 = stablehlo.transpose %83, dims = [0, 2, 1, 3] : (tensor<4x1024x32x64xbf16>) -> tensor<4x32x1024x64xbf16>
    %85 = stablehlo.convert %84 : (tensor<4x32x1024x64xbf16>) -> tensor<4x32x1024x64xf32>
    %86 = stablehlo.transpose %85, dims = [0, 2, 1, 3] : (tensor<4x32x1024x64xf32>) -> tensor<4x1024x32x64xf32>
    %87 = stablehlo.reshape %86 : (tensor<4x1024x32x64xf32>) -> tensor<8388608xf32>
    %88 = stablehlo.reshape %87 : (tensor<8388608xf32>) -> tensor<4x1024x32x64xf32>
    %89 = stablehlo.transpose %88, dims = [0, 2, 1, 3] : (tensor<4x1024x32x64xf32>) -> tensor<4x32x1024x64xf32>
    %90 = stablehlo.convert %arg0 : (tensor<f64>) -> tensor<f32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [] : (tensor<f32>) -> tensor<4x32x1024x64xf32>
    %92 = stablehlo.multiply %89, %91 : tensor<4x32x1024x64xf32>
    %93 = stablehlo.transpose %92, dims = [0, 2, 1, 3] : (tensor<4x32x1024x64xf32>) -> tensor<4x1024x32x64xf32>
    %94 = stablehlo.reshape %93 : (tensor<4x1024x32x64xf32>) -> tensor<8388608xf32>
    %95 = stablehlo.reshape %94 : (tensor<8388608xf32>) -> tensor<4x1024x32x64xf32>
    %96 = stablehlo.transpose %95, dims = [0, 2, 1, 3] : (tensor<4x1024x32x64xf32>) -> tensor<4x32x1024x64xf32>
    // CHECK: stablehlo.dot_general
    %97 = stablehlo.dot_general %96, %53, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [HIGHEST, HIGHEST] : (tensor<4x32x1024x64xf32>, tensor<4x32x64x1024xf32>) -> tensor<4x32x1024x1024xf32>
    %98 = stablehlo.broadcast_in_dim %10, dims = [2, 3] : (tensor<1024x1024xf32>) -> tensor<4x32x1024x1024xf32>
    %99 = stablehlo.add %97, %98 : tensor<4x32x1024x1024xf32>
    return %99 : tensor<4x32x1024x1024xf32>
  }
}