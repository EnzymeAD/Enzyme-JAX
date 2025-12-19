// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<4> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %1:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %cst) : tensor<i64>, tensor<4x4xf32>
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c : tensor<i32>
      %5:2 = stablehlo.while(%iterArg_4 = %c_0, %iterArg_5 = %iterArg_3) : tensor<i64>, tensor<4x4xf32>
      cond {
        %6 = stablehlo.compare  LT, %iterArg_4, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %6 : tensor<i1>
      } do {
        %6 = stablehlo.add %c_2, %iterArg_4 : tensor<i64>
        %7 = stablehlo.convert %6 : (tensor<i64>) -> tensor<i32>
        %8 = stablehlo.subtract %7, %c : tensor<i32>
        %9 = stablehlo.dynamic_slice %0, %4, %8, sizes = [1, 1] : (tensor<4x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %10 = stablehlo.dynamic_update_slice %iterArg_5, %9, %4, %8 : (tensor<4x4xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
        stablehlo.return %6, %10 : tensor<i64>, tensor<4x4xf32>
      }
      stablehlo.return %2, %5#1 : tensor<i64>, tensor<4x4xf32>
    }
    return %1#1 : tensor<4x4xf32>
}

// CHECK: func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:     return %0 : tensor<4x4xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<6x3x2x5x4x4xf32> {enzymexla.memory_effects = []}) -> tensor<6x3x5x5x2x2xf32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<3.000000e+00> : tensor<3x3x2x30xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<6x3x5x5x2x2xf32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %c_2 = stablehlo.constant dense<6> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %c_4 = stablehlo.constant dense<5> : tensor<i64>
    %c_5 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [3, 5, 4, 0, 2, 1] : (tensor<6x3x2x5x4x4xf32>) -> tensor<5x4x4x6x2x3xf32>
    %1:2 = stablehlo.while(%iterArg = %c_3, %iterArg_6 = %cst_0) : tensor<i64>, tensor<6x3x5x5x2x2xf32> attributes {enzyme.disable_mincut}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_5, %iterArg : tensor<i64>
      %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c_1 : tensor<i32>
      %5 = stablehlo.dynamic_slice %0, %iterArg, %c_3, %c_3, %c_3, %c_3, %c_3, sizes = [1, 4, 4, 6, 2, 3] : (tensor<5x4x4x6x2x3xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x4x4x6x2x3xf32>
      %6 = stablehlo.reshape %5 : (tensor<1x4x4x6x2x3xf32>) -> tensor<4x4x12x3xf32>
      %7 = stablehlo.convolution(%6, %cst) dim_numbers = [0, 1, f, b]x[0, 1, i, o]->[0, 1, f, b], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 6 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<4x4x12x3xf32>, tensor<3x3x2x30xf32>) -> tensor<2x2x30x3xf32>
      %8 = stablehlo.reshape %7 : (tensor<2x2x30x3xf32>) -> tensor<2x2x6x5x3xf32>
      %9 = stablehlo.transpose %8, dims = [2, 0, 1, 3, 4] : (tensor<2x2x6x5x3xf32>) -> tensor<6x2x2x5x3xf32>
      %10:2 = stablehlo.while(%iterArg_7 = %c_3, %iterArg_8 = %iterArg_6) : tensor<i64>, tensor<6x3x5x5x2x2xf32> attributes {enzyme.disable_mincut}
      cond {
        %11 = stablehlo.compare  LT, %iterArg_7, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %11 : tensor<i1>
      } do {
        %11 = stablehlo.add %c_5, %iterArg_7 : tensor<i64>
        %12 = stablehlo.convert %11 : (tensor<i64>) -> tensor<i32>
        %13 = stablehlo.subtract %12, %c_1 : tensor<i32>
        %14 = stablehlo.dynamic_slice %9, %iterArg_7, %c_3, %c_3, %c_3, %c_3, sizes = [1, 2, 2, 5, 3] : (tensor<6x2x2x5x3xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x2x2x5x3xf32>
        %15 = stablehlo.reshape %14 : (tensor<1x2x2x5x3xf32>) -> tensor<2x2x1x5x3x1xf32>
        %16 = stablehlo.transpose %15, dims = [5, 4, 3, 2, 1, 0] : (tensor<2x2x1x5x3x1xf32>) -> tensor<1x3x5x1x2x2xf32>
        %17 = stablehlo.dynamic_update_slice %iterArg_8, %16, %13, %c, %c, %4, %c, %c : (tensor<6x3x5x5x2x2xf32>, tensor<1x3x5x1x2x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x3x5x5x2x2xf32>
        stablehlo.return %11, %17 : tensor<i64>, tensor<6x3x5x5x2x2xf32>
      }
      stablehlo.return %2, %10#1 : tensor<i64>, tensor<6x3x5x5x2x2xf32>
    }
    return %1#1 : tensor<6x3x5x5x2x2xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<6x3x2x5x4x4xf32> {enzymexla.memory_effects = []}) -> tensor<6x3x5x5x2x2xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:   %cst = stablehlo.constant dense<3.000000e+00> : tensor<3x3x2x30xf32>
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<6x3x5x5x2x2xf32>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:   %c_4 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [3, 5, 4, 0, 2, 1] : (tensor<6x3x2x5x4x4xf32>) -> tensor<5x4x4x6x2x3xf32>
// CHECK-NEXT:   %1:2 = stablehlo.while(%iterArg = %c_2, %iterArg_5 = %cst_0) : tensor<i64>, tensor<6x3x5x5x2x2xf32> attributes {enzyme.disable_mincut}
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %2 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %2 = stablehlo.add %c_4, %iterArg {enzymexla.bounds = {{\[}}[1, 5]{{\]}}} : tensor<i64>
// CHECK-NEXT:     %3 = stablehlo.convert %2 {enzymexla.bounds = {{\[}}[1, 5]{{\]}}} : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:     %4 = stablehlo.subtract %3, %c_1 {enzymexla.bounds = {{\[}}[0, 4]{{\]}}} : tensor<i32>
// CHECK-NEXT:     %5 = stablehlo.dynamic_slice %0, %iterArg, %c_2, %c_2, %c_2, %c_2, %c_2, sizes = [1, 4, 4, 6, 2, 3] : (tensor<5x4x4x6x2x3xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x4x4x6x2x3xf32>
// CHECK-NEXT:     %6 = stablehlo.reshape %5 : (tensor<1x4x4x6x2x3xf32>) -> tensor<4x4x12x3xf32>
// CHECK-NEXT{LITERAL}:     %7 = stablehlo.convolution(%6, %cst) dim_numbers = [0, 1, f, b]x[0, 1, i, o]->[0, 1, f, b], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 6 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<4x4x12x3xf32>, tensor<3x3x2x30xf32>) -> tensor<2x2x30x3xf32>
// CHECK-NEXT:     %8 = stablehlo.reshape %7 : (tensor<2x2x30x3xf32>) -> tensor<2x2x6x5x3xf32>
// CHECK-NEXT:     %9 = stablehlo.broadcast_in_dim %8, dims = [5, 4, 0, 2, 1] : (tensor<2x2x6x5x3xf32>) -> tensor<6x3x5x1x2x2xf32>
// CHECK-NEXT:     %10 = stablehlo.dynamic_update_slice %iterArg_5, %9, %c, %c, %c, %4, %c, %c : (tensor<6x3x5x5x2x2xf32>, tensor<6x3x5x1x2x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x3x5x5x2x2xf32>
// CHECK-NEXT:     stablehlo.return %2, %10 : tensor<i64>, tensor<6x3x5x5x2x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %1#1 : tensor<6x3x5x5x2x2xf32>
// CHECK-NEXT: }
