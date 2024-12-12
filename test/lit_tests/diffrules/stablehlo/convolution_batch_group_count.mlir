// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%arg0: tensor<8x4x4x4xf32>, %arg1: tensor<3x3x4x4xf32>) -> tensor<2x4x2x2xf32> {
    %2 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 1, 0]x[0, 1, i, o]->[b, f, 1, 0], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {feature_group_count = 1 : i64, batch_group_count = 4 : i64} : (tensor<8x4x4x4xf32>, tensor<3x3x4x4xf32>) -> tensor<2x4x2x2xf32>
    return %2 : tensor<2x4x2x2xf32>
  }
}

// REVERSE{LITERAL}:  func.func @main(%arg0: tensor<8x4x4x4xf32>, %arg1: tensor<3x3x4x4xf32>, %arg2: tensor<2x4x2x2xf32>) -> (tensor<8x4x4x4xf32>, tensor<3x3x4x4xf32>) {
// REVERSE-NEXT{LITERAL}:    %0 = stablehlo.reshape %arg1 : (tensor<3x3x4x4xf32>) -> tensor<3x3x4x4x1xf32>
// REVERSE-NEXT{LITERAL}:    %1 = stablehlo.transpose %0, dims = [0, 1, 3, 2, 4] : (tensor<3x3x4x4x1xf32>) -> tensor<3x3x4x4x1xf32>
// REVERSE-NEXT{LITERAL}:    %2 = stablehlo.reshape %1 : (tensor<3x3x4x4x1xf32>) -> tensor<3x3x16x1xf32>
// REVERSE-NEXT{LITERAL}:    %3 = stablehlo.convolution(%arg2, %2) dim_numbers = [b, f, 1, 0]x[0, 1, o, i]->[b, f, 1, 0], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [true, true]} {batch_group_count = 1 : i64, feature_group_count = 4 : i64} : (tensor<2x4x2x2xf32>, tensor<3x3x16x1xf32>) -> tensor<2x16x4x4xf32>
// REVERSE-NEXT{LITERAL}:    %4 = stablehlo.reshape %3 : (tensor<2x16x4x4xf32>) -> tensor<2x4x4x4x4xf32>
// REVERSE-NEXT{LITERAL}:    %5 = stablehlo.transpose %4, dims = [1, 0, 2, 3, 4] : (tensor<2x4x4x4x4xf32>) -> tensor<4x2x4x4x4xf32>
// REVERSE-NEXT{LITERAL}:    %6 = stablehlo.reshape %5 : (tensor<4x2x4x4x4xf32>) -> tensor<8x4x4x4xf32>
// REVERSE-NEXT{LITERAL}:    %7 = stablehlo.convolution(%arg0, %arg2) dim_numbers = [f, b, 1, 0]x[i, o, 1, 0]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 4 : i64} : (tensor<8x4x4x4xf32>, tensor<2x4x2x2xf32>) -> tensor<3x3x4x4xf32>
// REVERSE-NEXT{LITERAL}:    return %6, %7 : tensor<8x4x4x4xf32>, tensor<3x3x4x4xf32>
// REVERSE-NEXT{LITERAL}:  }
