// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup,enzyme_dup retTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%arg0: tensor<3x1x4x4xf64>, %arg1: tensor<2x2x1x2xf64>) -> tensor<3x2x6x6xf64> {
    %2 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 1, 0]x[0, 1, i, o]->[b, f, 1, 0], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x1x4x4xf64>, tensor<2x2x1x2xf64>) -> tensor<3x2x6x6xf64>
    return %2 : tensor<3x2x6x6xf64>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<3x1x4x4xf64>, %arg1: tensor<3x1x4x4xf64>, %arg2: tensor<2x2x1x2xf64>, %arg3: tensor<2x2x1x2xf64>) -> (tensor<3x2x6x6xf64>, tensor<3x2x6x6xf64>) {
// FORWARD-NEXT{LITERAL}:    %0 = stablehlo.convolution(%arg1, %arg2) dim_numbers = [b, f, 1, 0]x[0, 1, i, o]->[b, f, 1, 0], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x1x4x4xf64>, tensor<2x2x1x2xf64>) -> tensor<3x2x6x6xf64>
// FORWARD-NEXT{LITERAL}:    %1 = stablehlo.convolution(%arg0, %arg3) dim_numbers = [b, f, 1, 0]x[0, 1, i, o]->[b, f, 1, 0], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x1x4x4xf64>, tensor<2x2x1x2xf64>) -> tensor<3x2x6x6xf64>
// FORWARD-NEXT{LITERAL}:    %2 = stablehlo.add %0, %1 : tensor<3x2x6x6xf64>
// FORWARD-NEXT{LITERAL}:    %3 = stablehlo.convolution(%arg0, %arg2) dim_numbers = [b, f, 1, 0]x[0, 1, i, o]->[b, f, 1, 0], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x1x4x4xf64>, tensor<2x2x1x2xf64>) -> tensor<3x2x6x6xf64>
// FORWARD-NEXT{LITERAL}:    return %3, %2 : tensor<3x2x6x6xf64>, tensor<3x2x6x6xf64>
// FORWARD-NEXT{LITERAL}:  }

// REVERSE:  func.func @main(%arg0: tensor<3x1x4x4xf64>, %arg1: tensor<2x2x1x2xf64>, %arg2: tensor<3x2x6x6xf64>) -> (tensor<3x1x4x4xf64>, tensor<2x2x1x2xf64>) {
// REVERSE-NEXT{LITERAL}:    %0 = stablehlo.convolution(%arg2, %arg1) dim_numbers = [b, f, 1, 0]x[0, 1, o, i]->[b, f, 1, 0], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [true, true]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x2x6x6xf64>, tensor<2x2x1x2xf64>) -> tensor<3x1x4x4xf64>
// REVERSE-NEXT{LITERAL}:    %1 = stablehlo.convolution(%arg0, %arg2) dim_numbers = [f, b, 1, 0]x[i, o, 1, 0]->[0, 1, b, f], window = {stride = [2, 2], pad = [[2, 2], [2, 2]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x1x4x4xf64>, tensor<3x2x6x6xf64>) -> tensor<2x2x1x2xf64>
// REVERSE-NEXT{LITERAL}:    return %0, %1 : tensor<3x1x4x4xf64>, tensor<2x2x1x2xf64>
// REVERSE-NEXT{LITERAL}:  }
