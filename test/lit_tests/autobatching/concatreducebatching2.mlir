// RUN: enzymexlamlir-opt --auto-batching %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<3x64xf32> {enzymexla.memory_effects = []}, %arg1: tensor<10x1xf32> {enzymexla.memory_effects = []}, %arg2: tensor<64x32xf32> {enzymexla.memory_effects = []}, %arg3: tensor<32xf32> {enzymexla.memory_effects = []}, %arg4: tensor<32x32xf32> {enzymexla.memory_effects = []}, %arg5: tensor<32xf32> {enzymexla.memory_effects = []}, %arg6: tensor<32x16xf32> {enzymexla.memory_effects = []}, %arg7: tensor<16xf32> {enzymexla.memory_effects = []}, %arg8: tensor<1x32xf32> {enzymexla.memory_effects = []}, %arg9: tensor<32xf32> {enzymexla.memory_effects = []}, %arg10: tensor<32x32xf32> {enzymexla.memory_effects = []}, %arg11: tensor<32xf32> {enzymexla.memory_effects = []}, %arg12: tensor<32x16xf32> {enzymexla.memory_effects = []}, %arg13: tensor<16xf32> {enzymexla.memory_effects = []}, %arg14: tensor<64x32xf32> {enzymexla.memory_effects = []}, %arg15: tensor<32xf32> {enzymexla.memory_effects = []}, %arg16: tensor<1x32xf32> {enzymexla.memory_effects = []}, %arg17: tensor<32xf32> {enzymexla.memory_effects = []}) -> tensor<3x10xf32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<3x32x10xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<32xf32>
    %0 = stablehlo.broadcast_in_dim %arg17, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %1 = stablehlo.dot_general %arg16, %arg1, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x32xf32>, tensor<10x1xf32>) -> tensor<32x10xf32>
    %2 = stablehlo.add %1, %0 : tensor<32x10xf32>
    %3 = stablehlo.slice %arg0 [0:1, 0:64] : (tensor<3x64xf32>) -> tensor<1x64xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x64xf32>) -> tensor<64xf32>
    %5 = stablehlo.dot_general %arg14, %4, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf32>, tensor<64xf32>) -> tensor<32xf32>
    %6 = stablehlo.add %5, %arg15 : tensor<32xf32>
    %7 = stablehlo.dot_general %arg8, %arg1, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x32xf32>, tensor<10x1xf32>) -> tensor<32x10xf32>
    %8 = stablehlo.broadcast_in_dim %arg9, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %9 = stablehlo.add %7, %8 : tensor<32x10xf32>
    %10 = stablehlo.tanh %9 : tensor<32x10xf32>
    %11 = stablehlo.dot_general %arg2, %4, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf32>, tensor<64xf32>) -> tensor<32xf32>
    %12 = stablehlo.add %11, %arg3 : tensor<32xf32>
    %13 = stablehlo.tanh %12 : tensor<32xf32>
    %14 = stablehlo.dot_general %arg4, %13, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %15 = stablehlo.add %14, %arg5 : tensor<32xf32>
    %16 = stablehlo.tanh %15 : tensor<32xf32>
    %17 = stablehlo.dot_general %arg10, %10, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x10xf32>) -> tensor<32x10xf32>
    %18 = stablehlo.subtract %cst_0, %16 : tensor<32xf32>
    %19 = stablehlo.multiply %18, %6 : tensor<32xf32>
    %20 = stablehlo.slice %arg0 [1:2, 0:64] : (tensor<3x64xf32>) -> tensor<1x64xf32>
    %21 = stablehlo.reshape %20 : (tensor<1x64xf32>) -> tensor<64xf32>
    %22 = stablehlo.dot_general %arg14, %21, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf32>, tensor<64xf32>) -> tensor<32xf32>
    %23 = stablehlo.add %22, %arg15 : tensor<32xf32>
    %24 = stablehlo.tanh %9 : tensor<32x10xf32>
    %25 = stablehlo.dot_general %arg2, %21, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf32>, tensor<64xf32>) -> tensor<32xf32>
    %26 = stablehlo.add %25, %arg3 : tensor<32xf32>
    %27 = stablehlo.tanh %26 : tensor<32xf32>
    %28 = stablehlo.dot_general %arg4, %27, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %29 = stablehlo.add %28, %arg5 : tensor<32xf32>
    %30 = stablehlo.tanh %29 : tensor<32xf32>
    %31 = stablehlo.dot_general %arg10, %24, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x10xf32>) -> tensor<32x10xf32>
    %32 = stablehlo.subtract %cst_0, %30 : tensor<32xf32>
    %33 = stablehlo.multiply %32, %23 : tensor<32xf32>
    %34 = stablehlo.slice %arg0 [2:3, 0:64] : (tensor<3x64xf32>) -> tensor<1x64xf32>
    %35 = stablehlo.reshape %34 : (tensor<1x64xf32>) -> tensor<64xf32>
    %36 = stablehlo.dot_general %arg14, %35, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf32>, tensor<64xf32>) -> tensor<32xf32>
    %37 = stablehlo.add %36, %arg15 : tensor<32xf32>
    %38 = stablehlo.tanh %9 : tensor<32x10xf32>
    %39 = stablehlo.dot_general %arg2, %35, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf32>, tensor<64xf32>) -> tensor<32xf32>
    %40 = stablehlo.add %39, %arg3 : tensor<32xf32>
    %41 = stablehlo.tanh %40 : tensor<32xf32>
    %42 = stablehlo.dot_general %arg4, %41, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %43 = stablehlo.add %42, %arg5 : tensor<32xf32>
    %44 = stablehlo.tanh %43 : tensor<32xf32>
    %45 = stablehlo.dot_general %arg10, %38, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x10xf32>) -> tensor<32x10xf32>
    %46 = stablehlo.subtract %cst_0, %44 : tensor<32xf32>
    %47 = stablehlo.multiply %46, %37 : tensor<32xf32>
    %48 = stablehlo.broadcast_in_dim %19, dims = [1] : (tensor<32xf32>) -> tensor<1x32x10xf32>
    %49 = stablehlo.broadcast_in_dim %33, dims = [1] : (tensor<32xf32>) -> tensor<1x32x10xf32>
    %50 = stablehlo.broadcast_in_dim %47, dims = [1] : (tensor<32xf32>) -> tensor<1x32x10xf32>
    %51 = stablehlo.concatenate %48, %49, %50, dim = 0 : (tensor<1x32x10xf32>, tensor<1x32x10xf32>, tensor<1x32x10xf32>) -> tensor<3x32x10xf32>
    %52 = stablehlo.broadcast_in_dim %16, dims = [1] : (tensor<32xf32>) -> tensor<1x32x10xf32>
    %53 = stablehlo.broadcast_in_dim %30, dims = [1] : (tensor<32xf32>) -> tensor<1x32x10xf32>
    %54 = stablehlo.broadcast_in_dim %44, dims = [1] : (tensor<32xf32>) -> tensor<1x32x10xf32>
    %55 = stablehlo.concatenate %52, %53, %54, dim = 0 : (tensor<1x32x10xf32>, tensor<1x32x10xf32>, tensor<1x32x10xf32>) -> tensor<3x32x10xf32>
    %56 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<32x10xf32>) -> tensor<3x32x10xf32>
    %57 = stablehlo.multiply %55, %56 : tensor<3x32x10xf32>
    %58 = stablehlo.add %51, %57 : tensor<3x32x10xf32>
    %59 = stablehlo.dot_general %arg6, %58, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<3x32x10xf32>) -> tensor<16x3x10xf32>
    %60 = stablehlo.transpose %59, dims = [1, 0, 2] : (tensor<16x3x10xf32>) -> tensor<3x16x10xf32>
    %61 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<16xf32>) -> tensor<3x16x10xf32>
    %62 = stablehlo.add %60, %61 : tensor<3x16x10xf32>
    %63 = stablehlo.reshape %17 : (tensor<32x10xf32>) -> tensor<1x32x10xf32>
    %64 = stablehlo.reshape %31 : (tensor<32x10xf32>) -> tensor<1x32x10xf32>
    %65 = stablehlo.reshape %45 : (tensor<32x10xf32>) -> tensor<1x32x10xf32>
    %66 = stablehlo.concatenate %63, %64, %65, dim = 0 : (tensor<1x32x10xf32>, tensor<1x32x10xf32>, tensor<1x32x10xf32>) -> tensor<3x32x10xf32>
    %67 = stablehlo.broadcast_in_dim %arg11, dims = [1] : (tensor<32xf32>) -> tensor<3x32x10xf32>
    %68 = stablehlo.add %66, %67 : tensor<3x32x10xf32>
    %69 = stablehlo.tanh %68 : tensor<3x32x10xf32>
    %70 = stablehlo.subtract %cst, %69 : tensor<3x32x10xf32>
    %71 = stablehlo.broadcast_in_dim %6, dims = [1] : (tensor<32xf32>) -> tensor<1x32x10xf32>
    %72 = stablehlo.broadcast_in_dim %23, dims = [1] : (tensor<32xf32>) -> tensor<1x32x10xf32>
    %73 = stablehlo.broadcast_in_dim %37, dims = [1] : (tensor<32xf32>) -> tensor<1x32x10xf32>
    %74 = stablehlo.concatenate %71, %72, %73, dim = 0 : (tensor<1x32x10xf32>, tensor<1x32x10xf32>, tensor<1x32x10xf32>) -> tensor<3x32x10xf32>
    %75 = stablehlo.multiply %70, %74 : tensor<3x32x10xf32>
    %76 = stablehlo.multiply %69, %56 : tensor<3x32x10xf32>
    %77 = stablehlo.add %75, %76 : tensor<3x32x10xf32>
    %78 = stablehlo.dot_general %arg12, %77, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<3x32x10xf32>) -> tensor<16x3x10xf32>
    %79 = stablehlo.transpose %78, dims = [1, 0, 2] : (tensor<16x3x10xf32>) -> tensor<3x16x10xf32>
    %80 = stablehlo.broadcast_in_dim %arg13, dims = [1] : (tensor<16xf32>) -> tensor<3x16x10xf32>
    %81 = stablehlo.add %79, %80 : tensor<3x16x10xf32>
    %82 = stablehlo.dot_general %62, %81, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [1] : (tensor<3x16x10xf32>, tensor<3x16x10xf32>) -> tensor<3x10xf32>
    return %82 : tensor<3x10xf32>
  }
}
