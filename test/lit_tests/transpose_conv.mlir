// RUN: enzymexlamlir-opt --pass-pipeline="any(enzyme-hlo-generate-td{patterns=transpose_convolution},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

func.func @main(%arg0: tensor<5x2x1024xf32>, %arg1: tensor<4x2x1xf32>, %arg2: tensor<4xf32>, %arg3: tensor<16x4x16xcomplex<f32>>) -> (tensor<4x5x1024xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<5x2x1024xf32>) -> tensor<1024x2x5xf32>
    %1 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<4x2x1xf32>) -> tensor<1x2x4xf32>
    %2 = stablehlo.reverse %1, dims = [0] : tensor<1x2x4xf32>
    %3 = stablehlo.convolution(%0, %2) dim_numbers = [0, f, b]x[0, i, o]->[0, f, b], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1024x2x5xf32>, tensor<1x2x4xf32>) -> tensor<1024x4x5xf32>
    %4 = stablehlo.transpose %3, dims = [1, 2, 0] : (tensor<1024x4x5xf32>) -> tensor<4x5x1024xf32>
    return %4 : tensor<4x5x1024xf32>
}

// CHECK: func.func @main(%arg0: tensor<5x2x1024xf32>, %arg1: tensor<4x2x1xf32>, %arg2: tensor<4xf32>, %arg3: tensor<16x4x16xcomplex<f32>>) -> tensor<4x5x1024xf32> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<5x2x1024xf32>) -> tensor<1024x2x5xf32>
// CHECK-NEXT:     %1 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<4x2x1xf32>) -> tensor<1x2x4xf32>
// CHECK-NEXT:     %2 = stablehlo.reverse %1, dims = [0] : tensor<1x2x4xf32>
// CHECK-NEXT{LITERAL}:     %3 = stablehlo.convolution(%0, %2) dim_numbers = [0, f, b]x[0, i, o]->[f, b, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1024x2x5xf32>, tensor<1x2x4xf32>) -> tensor<4x5x1024xf32>
// CHECK-NEXT:     return %3 : tensor<4x5x1024xf32>
// CHECK-NEXT: }
