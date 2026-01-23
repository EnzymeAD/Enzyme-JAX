// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<3x64xf32> {enzymexla.memory_effects = []}, %arg1: tensor<10x1xf32> {enzymexla.memory_effects = []}, %arg2: tensor<64x32xf32> {enzymexla.memory_effects = []}, %arg3: tensor<32xf32> {enzymexla.memory_effects = []}, %arg4: tensor<32x32xf32> {enzymexla.memory_effects = []}, %arg5: tensor<32xf32> {enzymexla.memory_effects = []}, %arg6: tensor<32x16xf32> {enzymexla.memory_effects = []}, %arg7: tensor<16xf32> {enzymexla.memory_effects = []}, %arg8: tensor<1x32xf32> {enzymexla.memory_effects = []}, %arg9: tensor<32xf32> {enzymexla.memory_effects = []}, %arg10: tensor<32x32xf32> {enzymexla.memory_effects = []}, %arg11: tensor<32xf32> {enzymexla.memory_effects = []}, %arg12: tensor<32x16xf32> {enzymexla.memory_effects = []}, %arg13: tensor<16xf32> {enzymexla.memory_effects = []}, %arg14: tensor<64x32xf32> {enzymexla.memory_effects = []}, %arg15: tensor<32xf32> {enzymexla.memory_effects = []}, %arg16: tensor<1x32xf32> {enzymexla.memory_effects = []}, %arg17: tensor<32xf32> {enzymexla.memory_effects = []}) -> tensor<3x10xf32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<32x10xf32>
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
    %18 = stablehlo.broadcast_in_dim %arg11, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %19 = stablehlo.add %17, %18 : tensor<32x10xf32>
    %20 = stablehlo.tanh %19 : tensor<32x10xf32>
    %21 = stablehlo.subtract %cst_0, %16 : tensor<32xf32>
    %22 = stablehlo.multiply %21, %6 : tensor<32xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %24 = stablehlo.broadcast_in_dim %16, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %25 = stablehlo.multiply %24, %2 : tensor<32x10xf32>
    %26 = stablehlo.add %23, %25 : tensor<32x10xf32>
    %27 = stablehlo.subtract %cst, %20 : tensor<32x10xf32>
    %28 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %29 = stablehlo.multiply %27, %28 : tensor<32x10xf32>
    %30 = stablehlo.multiply %20, %2 : tensor<32x10xf32>
    %31 = stablehlo.add %29, %30 : tensor<32x10xf32>
    %32 = stablehlo.broadcast_in_dim %arg7, dims = [0] : (tensor<16xf32>) -> tensor<16x10xf32>
    %33 = stablehlo.dot_general %arg6, %26, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<32x10xf32>) -> tensor<16x10xf32>
    %34 = stablehlo.add %33, %32 : tensor<16x10xf32>
    %35 = stablehlo.broadcast_in_dim %arg13, dims = [0] : (tensor<16xf32>) -> tensor<16x10xf32>
    %36 = stablehlo.dot_general %arg12, %31, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<32x10xf32>) -> tensor<16x10xf32>
    %37 = stablehlo.add %36, %35 : tensor<16x10xf32>
    %38 = stablehlo.dot_general %34, %37, batching_dims = [1] x [1], contracting_dims = [0] x [0] : (tensor<16x10xf32>, tensor<16x10xf32>) -> tensor<10xf32>
    %39 = stablehlo.reshape %38 : (tensor<10xf32>) -> tensor<10x1xf32>
    %40 = stablehlo.slice %arg0 [1:2, 0:64] : (tensor<3x64xf32>) -> tensor<1x64xf32>
    %41 = stablehlo.reshape %40 : (tensor<1x64xf32>) -> tensor<64xf32>
    %42 = stablehlo.dot_general %arg14, %41, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf32>, tensor<64xf32>) -> tensor<32xf32>
    %43 = stablehlo.add %42, %arg15 : tensor<32xf32>
    %44 = stablehlo.tanh %9 : tensor<32x10xf32>
    %45 = stablehlo.dot_general %arg2, %41, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf32>, tensor<64xf32>) -> tensor<32xf32>
    %46 = stablehlo.add %45, %arg3 : tensor<32xf32>
    %47 = stablehlo.tanh %46 : tensor<32xf32>
    %48 = stablehlo.dot_general %arg4, %47, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %49 = stablehlo.add %48, %arg5 : tensor<32xf32>
    %50 = stablehlo.tanh %49 : tensor<32xf32>
    %51 = stablehlo.dot_general %arg10, %44, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x10xf32>) -> tensor<32x10xf32>
    %52 = stablehlo.add %51, %18 : tensor<32x10xf32>
    %53 = stablehlo.tanh %52 : tensor<32x10xf32>
    %54 = stablehlo.subtract %cst_0, %50 : tensor<32xf32>
    %55 = stablehlo.multiply %54, %43 : tensor<32xf32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %57 = stablehlo.broadcast_in_dim %50, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %58 = stablehlo.multiply %57, %2 : tensor<32x10xf32>
    %59 = stablehlo.add %56, %58 : tensor<32x10xf32>
    %60 = stablehlo.subtract %cst, %53 : tensor<32x10xf32>
    %61 = stablehlo.broadcast_in_dim %43, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %62 = stablehlo.multiply %60, %61 : tensor<32x10xf32>
    %63 = stablehlo.multiply %53, %2 : tensor<32x10xf32>
    %64 = stablehlo.add %62, %63 : tensor<32x10xf32>
    %65 = stablehlo.dot_general %arg6, %59, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<32x10xf32>) -> tensor<16x10xf32>
    %66 = stablehlo.add %65, %32 : tensor<16x10xf32>
    %67 = stablehlo.dot_general %arg12, %64, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<32x10xf32>) -> tensor<16x10xf32>
    %68 = stablehlo.add %67, %35 : tensor<16x10xf32>
    %69 = stablehlo.dot_general %66, %68, batching_dims = [1] x [1], contracting_dims = [0] x [0] : (tensor<16x10xf32>, tensor<16x10xf32>) -> tensor<10xf32>
    %70 = stablehlo.reshape %69 : (tensor<10xf32>) -> tensor<10x1xf32>
    %71 = stablehlo.slice %arg0 [2:3, 0:64] : (tensor<3x64xf32>) -> tensor<1x64xf32>
    %72 = stablehlo.reshape %71 : (tensor<1x64xf32>) -> tensor<64xf32>
    %73 = stablehlo.dot_general %arg14, %72, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf32>, tensor<64xf32>) -> tensor<32xf32>
    %74 = stablehlo.add %73, %arg15 : tensor<32xf32>
    %75 = stablehlo.tanh %9 : tensor<32x10xf32>
    %76 = stablehlo.dot_general %arg2, %72, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf32>, tensor<64xf32>) -> tensor<32xf32>
    %77 = stablehlo.add %76, %arg3 : tensor<32xf32>
    %78 = stablehlo.tanh %77 : tensor<32xf32>
    %79 = stablehlo.dot_general %arg4, %78, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %80 = stablehlo.add %79, %arg5 : tensor<32xf32>
    %81 = stablehlo.tanh %80 : tensor<32xf32>
    %82 = stablehlo.dot_general %arg10, %75, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x10xf32>) -> tensor<32x10xf32>
    %83 = stablehlo.add %82, %18 : tensor<32x10xf32>
    %84 = stablehlo.tanh %83 : tensor<32x10xf32>
    %85 = stablehlo.subtract %cst_0, %81 : tensor<32xf32>
    %86 = stablehlo.multiply %85, %74 : tensor<32xf32>
    %87 = stablehlo.broadcast_in_dim %86, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %88 = stablehlo.broadcast_in_dim %81, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %89 = stablehlo.multiply %88, %2 : tensor<32x10xf32>
    %90 = stablehlo.add %87, %89 : tensor<32x10xf32>
    %91 = stablehlo.subtract %cst, %84 : tensor<32x10xf32>
    %92 = stablehlo.broadcast_in_dim %74, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %93 = stablehlo.multiply %91, %92 : tensor<32x10xf32>
    %94 = stablehlo.multiply %84, %2 : tensor<32x10xf32>
    %95 = stablehlo.add %93, %94 : tensor<32x10xf32>
    %96 = stablehlo.dot_general %arg6, %90, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<32x10xf32>) -> tensor<16x10xf32>
    %97 = stablehlo.add %96, %32 : tensor<16x10xf32>
    %98 = stablehlo.dot_general %arg12, %95, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<32x10xf32>) -> tensor<16x10xf32>
    %99 = stablehlo.add %98, %35 : tensor<16x10xf32>
    %100 = stablehlo.dot_general %97, %99, batching_dims = [1] x [1], contracting_dims = [0] x [0] : (tensor<16x10xf32>, tensor<16x10xf32>) -> tensor<10xf32>
    %101 = stablehlo.reshape %100 : (tensor<10xf32>) -> tensor<10x1xf32>
    %102 = stablehlo.concatenate %39, %70, %101, dim = 1 : (tensor<10x1xf32>, tensor<10x1xf32>, tensor<10x1xf32>) -> tensor<10x3xf32>
    %103 = stablehlo.transpose %102, dims = [1, 0] : (tensor<10x3xf32>) -> tensor<3x10xf32>
    return %103 : tensor<3x10xf32>
  }
}
