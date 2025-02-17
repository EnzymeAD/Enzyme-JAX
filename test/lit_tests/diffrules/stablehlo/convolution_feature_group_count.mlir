// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<3x3x1x8xf32>) -> tensor<1x8x2x2xf32> {
    %2 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 1, 0]x[0, 1, i, o]->[b, f, 1, 0], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 4 : i64} : (tensor<1x4x4x4xf32>, tensor<3x3x1x8xf32>) -> tensor<1x8x2x2xf32>
    return %2 : tensor<1x8x2x2xf32>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<1x4x4x4xf32>, %arg2: tensor<3x3x1x8xf32>, %arg3: tensor<3x3x1x8xf32>) -> (tensor<1x8x2x2xf32>, tensor<1x8x2x2xf32>) {
// FORWARD-NEXT{LITERAL}:    %0 = stablehlo.convolution(%arg1, %arg2) dim_numbers = [b, f, 1, 0]x[0, 1, i, o]->[b, f, 1, 0], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 4 : i64} : (tensor<1x4x4x4xf32>, tensor<3x3x1x8xf32>) -> tensor<1x8x2x2xf32>
// FORWARD-NEXT{LITERAL}:    %1 = stablehlo.convolution(%arg0, %arg3) dim_numbers = [b, f, 1, 0]x[0, 1, i, o]->[b, f, 1, 0], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 4 : i64} : (tensor<1x4x4x4xf32>, tensor<3x3x1x8xf32>) -> tensor<1x8x2x2xf32>
// FORWARD-NEXT{LITERAL}:    %2 = stablehlo.add %0, %1 : tensor<1x8x2x2xf32>
// FORWARD-NEXT{LITERAL}:    %3 = stablehlo.convolution(%arg0, %arg2) dim_numbers = [b, f, 1, 0]x[0, 1, i, o]->[b, f, 1, 0], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 4 : i64} : (tensor<1x4x4x4xf32>, tensor<3x3x1x8xf32>) -> tensor<1x8x2x2xf32>
// FORWARD-NEXT{LITERAL}:    return %3, %2 : tensor<1x8x2x2xf32>, tensor<1x8x2x2xf32>
// FORWARD-NEXT{LITERAL}:  }

// REVERSE{LITERAL}:  func.func @main(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<3x3x1x8xf32>, %arg2: tensor<1x8x2x2xf32>) -> (tensor<1x4x4x4xf32>, tensor<3x3x1x8xf32>) {
// REVERSE-NEXT{LITERAL}:    %0 = stablehlo.reshape %arg1 : (tensor<3x3x1x8xf32>) -> tensor<3x3x4x2xf32>
// REVERSE-NEXT{LITERAL}:    %1 = stablehlo.convolution(%arg2, %0) dim_numbers = [b, f, 1, 0]x[0, 1, o, i]->[b, f, 1, 0], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [true, true]} {batch_group_count = 1 : i64, feature_group_count = 4 : i64} : (tensor<1x8x2x2xf32>, tensor<3x3x4x2xf32>) -> tensor<1x4x4x4xf32>
// REVERSE-NEXT{LITERAL}:    %2 = stablehlo.convolution(%arg0, %arg2) dim_numbers = [f, b, 1, 0]x[i, o, 1, 0]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 4 : i64, feature_group_count = 1 : i64} : (tensor<1x4x4x4xf32>, tensor<1x8x2x2xf32>) -> tensor<3x3x1x8xf32>
// REVERSE-NEXT{LITERAL}:    return %1, %2 : tensor<1x4x4x4xf32>, tensor<3x3x1x8xf32>
// REVERSE-NEXT{LITERAL}:  }
