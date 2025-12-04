// RUN: enzymexlamlir-opt --enzyme-batch --inline --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func private @unbatched_conv_to_batch(%arg0: tensor<4x4x2x3xf32>) -> tensor<2x2x7x3xf32> {
    %cst = stablehlo.constant dense<3.000000e+00> : tensor<3x3x2x7xf32>
    %0 = stablehlo.convolution(%arg0, %cst) dim_numbers = [0, 1, f, b]x[0, 1, i, o]->[0, 1, f, b], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<4x4x2x3xf32>, tensor<3x3x2x7xf32>) -> tensor<2x2x7x3xf32>
    return %0 : tensor<2x2x7x3xf32>
  }
  func.func @main(%arg0: tensor<6x3x2x5x4x4xf32>) -> tensor<6x3x7x5x2x2xf32> {
    %0 = stablehlo.transpose %arg0, dims = [3, 0, 5, 4, 2, 1] : (tensor<6x3x2x5x4x4xf32>) -> tensor<5x6x4x4x2x3xf32>
    %1 = enzyme.batch @unbatched_conv_to_batch(%0) {batch_shape = array<i64: 5, 6>} : (tensor<5x6x4x4x2x3xf32>) -> tensor<5x6x2x2x7x3xf32>
    %2 = stablehlo.transpose %1, dims = [1, 5, 4, 0, 3, 2] : (tensor<5x6x2x2x7x3xf32>) -> tensor<6x3x7x5x2x2xf32>
    return %2 : tensor<6x3x7x5x2x2xf32>
  }

  // CHECK: func.func @main(%arg0: tensor<6x3x2x5x4x4xf32>) -> tensor<6x3x7x5x2x2xf32> {
  // CHECK-NEXT:    %cst = stablehlo.constant dense<3.000000e+00> : tensor<3x3x2x210xf32>
  // CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [5, 4, 3, 0, 2, 1] : (tensor<6x3x2x5x4x4xf32>) -> tensor<4x4x5x6x2x3xf32>
  // CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<4x4x5x6x2x3xf32>) -> tensor<4x4x60x3xf32>
  // CHECK-NEXT{LITERAL}:    %2 = stablehlo.convolution(%1, %cst) dim_numbers = [0, 1, f, b]x[0, 1, i, o]->[0, 1, f, b], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 30 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<4x4x60x3xf32>, tensor<3x3x2x210xf32>) -> tensor<2x2x210x3xf32>
  // CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<2x2x210x3xf32>) -> tensor<2x2x5x6x7x3xf32>
  // CHECK-NEXT:    %4 = stablehlo.transpose %3, dims = [3, 5, 4, 2, 1, 0] : (tensor<2x2x5x6x7x3xf32>) -> tensor<6x3x7x5x2x2xf32>
  // CHECK-NEXT:    return %4 : tensor<6x3x7x5x2x2xf32>
  // CHECK-NEXT:  }
}

module {
  func.func private @unbatched_conv_to_batch2(%arg0: tensor<4x4x2x3xf32>, %arg1: tensor<3x3x2x7xf32>) -> tensor<2x2x7x3xf32> {
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [0, 1, f, b]x[0, 1, i, o]->[0, 1, f, b], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<4x4x2x3xf32>, tensor<3x3x2x7xf32>) -> tensor<2x2x7x3xf32>
    return %0 : tensor<2x2x7x3xf32>
  }
  func.func @main(%arg0: tensor<6x3x2x5x4x4xf32>) -> tensor<6x3x7x5x2x2xf32> {
    %cst = stablehlo.constant dense<3.000000e+00> : tensor<5x6x3x3x2x7xf32>
    %0 = stablehlo.transpose %arg0, dims = [3, 0, 5, 4, 2, 1] : (tensor<6x3x2x5x4x4xf32>) -> tensor<5x6x4x4x2x3xf32>
    %1 = enzyme.batch @unbatched_conv_to_batch2(%0, %cst) {batch_shape = array<i64: 5, 6>} : (tensor<5x6x4x4x2x3xf32>, tensor<5x6x3x3x2x7xf32>) -> tensor<5x6x2x2x7x3xf32>
    %2 = stablehlo.transpose %1, dims = [1, 5, 4, 0, 3, 2] : (tensor<5x6x2x2x7x3xf32>) -> tensor<6x3x7x5x2x2xf32>
    return %2 : tensor<6x3x7x5x2x2xf32>
  }

  // CHECK: func.func @main(%arg0: tensor<6x3x2x5x4x4xf32>) -> tensor<6x3x7x5x2x2xf32> {
  // CHECK-NEXT:    %cst = stablehlo.constant dense<3.000000e+00> : tensor<3x3x2x210xf32>
  // CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [5, 4, 3, 0, 2, 1] : (tensor<6x3x2x5x4x4xf32>) -> tensor<4x4x5x6x2x3xf32>
  // CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<4x4x5x6x2x3xf32>) -> tensor<4x4x60x3xf32>
  // CHECK-NEXT{LITERAL}:    %2 = stablehlo.convolution(%1, %cst) dim_numbers = [0, 1, f, b]x[0, 1, i, o]->[0, 1, f, b], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 30 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<4x4x60x3xf32>, tensor<3x3x2x210xf32>) -> tensor<2x2x210x3xf32>
  // CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<2x2x210x3xf32>) -> tensor<2x2x5x6x7x3xf32>
  // CHECK-NEXT:    %4 = stablehlo.transpose %3, dims = [3, 5, 4, 2, 1, 0] : (tensor<2x2x5x6x7x3xf32>) -> tensor<6x3x7x5x2x2xf32>
  // CHECK-NEXT:    return %4 : tensor<6x3x7x5x2x2xf32>
  // CHECK-NEXT:  }
}

module {
  func.func private @unbatched_conv_to_batch3(%arg0: tensor<4x4x2x3xf32>) -> tensor<2x2x7x3xf32> {
    %iota = stablehlo.iota dim = 0 : tensor<3x3x2x7xi32>
    %w = stablehlo.convert %iota : (tensor<3x3x2x7xi32>) -> tensor<3x3x2x7xf32>
    %0 = stablehlo.convolution(%arg0, %w) dim_numbers = [0, 1, f, b]x[0, 1, i, o]->[0, 1, f, b], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<4x4x2x3xf32>, tensor<3x3x2x7xf32>) -> tensor<2x2x7x3xf32>
    return %0 : tensor<2x2x7x3xf32>
  }
  func.func @main(%arg0: tensor<6x3x2x5x4x4xf32>) -> tensor<6x3x7x5x2x2xf32> {
    %0 = stablehlo.transpose %arg0, dims = [3, 0, 5, 4, 2, 1] : (tensor<6x3x2x5x4x4xf32>) -> tensor<5x6x4x4x2x3xf32>
    %1 = enzyme.batch @unbatched_conv_to_batch3(%0) {batch_shape = array<i64: 5, 6>} : (tensor<5x6x4x4x2x3xf32>) -> tensor<5x6x2x2x7x3xf32>
    %2 = stablehlo.transpose %1, dims = [1, 5, 4, 0, 3, 2] : (tensor<5x6x2x2x7x3xf32>) -> tensor<6x3x7x5x2x2xf32>
    return %2 : tensor<6x3x7x5x2x2xf32>
  }

  // CHECK: func.func @main(%arg0: tensor<6x3x2x5x4x4xf32>) -> tensor<6x3x7x5x2x2xf32> {
  // CHECK-NEXT:     %0 = stablehlo.iota dim = 2 : tensor<5x6x3x3x2x7xi32>
  // CHECK-NEXT:     %1 = stablehlo.convert %0 : (tensor<5x6x3x3x2x7xi32>) -> tensor<5x6x3x3x2x7xf32>
  // CHECK-NEXT:     %2 = stablehlo.transpose %arg0, dims = [5, 4, 3, 0, 2, 1] : (tensor<6x3x2x5x4x4xf32>) -> tensor<4x4x5x6x2x3xf32>
  // CHECK-NEXT:     %3 = stablehlo.reshape %2 : (tensor<4x4x5x6x2x3xf32>) -> tensor<4x4x60x3xf32>
  // CHECK-NEXT:     %4 = stablehlo.transpose %1, dims = [2, 3, 4, 0, 1, 5] : (tensor<5x6x3x3x2x7xf32>) -> tensor<3x3x2x5x6x7xf32>
  // CHECK-NEXT:     %5 = stablehlo.reshape %4 : (tensor<3x3x2x5x6x7xf32>) -> tensor<3x3x2x210xf32>
  // CHECK-NEXT{LITERAL}:     %6 = stablehlo.convolution(%3, %5) dim_numbers = [0, 1, f, b]x[0, 1, i, o]->[0, 1, f, b], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 30 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<4x4x60x3xf32>, tensor<3x3x2x210xf32>) -> tensor<2x2x210x3xf32>
  // CHECK-NEXT:     %7 = stablehlo.reshape %6 : (tensor<2x2x210x3xf32>) -> tensor<2x2x5x6x7x3xf32>
  // CHECK-NEXT:     %8 = stablehlo.transpose %7, dims = [3, 5, 4, 2, 1, 0] : (tensor<2x2x5x6x7x3xf32>) -> tensor<6x3x7x5x2x2xf32>
  // CHECK-NEXT:     return %8 : tensor<6x3x7x5x2x2xf32>
  // CHECK-NEXT:   }
}
