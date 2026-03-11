// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-enzymexla)" | FileCheck %s

module {
  // Simple 1-D rank-1 stencil: input=[10], weights=[3], output=[8]
  // Semantics: output[i] = sum_{k=0..2} input[i+k] * weights[k]
  func.func @stencil_1d(%input: tensor<10xf32>, %weights: tensor<3xf32>) -> tensor<8xf32> {
    %result = enzymexla.stencil %input, %weights : (tensor<10xf32>, tensor<3xf32>) -> tensor<8xf32>
    return %result : tensor<8xf32>
  }

  // 2-D rank-2 stencil: input=[5x5], weights=[2x2], output=[4x4]
  // Semantics: output[i,j] = sum_{ki,kj} input[i+ki, j+kj] * weights[ki, kj]
  func.func @stencil_2d(%input: tensor<5x5xf32>, %weights: tensor<2x2xf32>) -> tensor<4x4xf32> {
    %result = enzymexla.stencil %input, %weights : (tensor<5x5xf32>, tensor<2x2xf32>) -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }

  // Rank-2 with f64 element type
  func.func @stencil_2d_f64(%input: tensor<8x8xf64>, %weights: tensor<3x3xf64>) -> tensor<6x6xf64> {
    %result = enzymexla.stencil %input, %weights : (tensor<8x8xf64>, tensor<3x3xf64>) -> tensor<6x6xf64>
    return %result : tensor<6x6xf64>
  }
}
