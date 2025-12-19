// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reverse_licm(0);reshape_licm(0);while_simplify(1);while_is_copy_simplify;greedy_while_loop_batch_fission" --transform-interpreter --enzyme-hlo-remove-transform  --enzyme-hlo-opt --auto-batching --enzyme-hlo-opt --enzyme-hlo-generate-td="patterns=reverse_licm(0);reshape_licm(0);while_simplify(1);while_is_copy_simplify;greedy_while_loop_batch_fission;transpose_reshape" --transform-interpreter --enzyme-hlo-remove-transform --enzyme-hlo-opt --auto-batching --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<6x3x2x5x4x4xf32> {enzymexla.memory_effects = []}) -> tensor<6x3x5x5x2x2xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<6x3x5x5x2x2xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<3x3x2x5xf32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %c_2 = stablehlo.constant dense<6> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %c_4 = stablehlo.constant dense<5> : tensor<i64>
    %c_5 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [5, 4, 3, 2, 1, 0] : (tensor<6x3x2x5x4x4xf32>) -> tensor<4x4x5x2x3x6xf32>
    %1:3 = stablehlo.while(%iterArg = %c_3, %iterArg_6 = %cst_0, %iterArg_7 = %cst) : tensor<i64>, tensor<3x3x2x5xf32>, tensor<6x3x5x5x2x2xf32> attributes {enzyme.disable_mincut}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_5, %iterArg : tensor<i64>
      %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c_1 : tensor<i32>
      %5:3 = stablehlo.while(%iterArg_8 = %c_3, %iterArg_9 = %iterArg_6, %iterArg_10 = %iterArg_7) : tensor<i64>, tensor<3x3x2x5xf32>, tensor<6x3x5x5x2x2xf32> attributes {enzyme.disable_mincut}
      cond {
        %6 = stablehlo.compare  LT, %iterArg_8, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %6 : tensor<i1>
      } do {
        %6 = stablehlo.add %c_5, %iterArg_8 : tensor<i64>
        %7 = stablehlo.convert %6 : (tensor<i64>) -> tensor<i32>
        %8 = stablehlo.subtract %7, %c_1 : tensor<i32>
        %9 = stablehlo.dynamic_slice %0, %c, %c, %4, %c, %c, %8, sizes = [4, 4, 1, 2, 3, 1] : (tensor<4x4x5x2x3x6xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x4x1x2x3x1xf32>
        %10 = stablehlo.reshape %9 : (tensor<4x4x1x2x3x1xf32>) -> tensor<4x4x2x3xf32>
        %11 = stablehlo.reverse %iterArg_9, dims = [0, 1] : tensor<3x3x2x5xf32>
        %12 = stablehlo.convolution(%10, %11) dim_numbers = [0, 1, f, b]x[0, 1, i, o]->[0, 1, f, b], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<4x4x2x3xf32>, tensor<3x3x2x5xf32>) -> tensor<2x2x5x3xf32>
        %13 = stablehlo.reshape %12 : (tensor<2x2x5x3xf32>) -> tensor<2x2x1x5x3x1xf32>
        %14 = stablehlo.transpose %13, dims = [5, 4, 3, 2, 1, 0] : (tensor<2x2x1x5x3x1xf32>) -> tensor<1x3x5x1x2x2xf32>
        %15 = stablehlo.dynamic_update_slice %iterArg_10, %14, %8, %c, %c, %4, %c, %c : (tensor<6x3x5x5x2x2xf32>, tensor<1x3x5x1x2x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x3x5x5x2x2xf32>
        stablehlo.return %6, %iterArg_9, %15 : tensor<i64>, tensor<3x3x2x5xf32>, tensor<6x3x5x5x2x2xf32>
      }
      stablehlo.return %2, %5#1, %5#2 : tensor<i64>, tensor<3x3x2x5xf32>, tensor<6x3x5x5x2x2xf32>
    }
    return %1#2 : tensor<6x3x5x5x2x2xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<6x3x2x5x4x4xf32> {enzymexla.memory_effects = []}) -> tensor<6x3x5x5x2x2xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     %cst = stablehlo.constant dense<3.000000e+00> : tensor<3x3x2x150xf32>
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [4, 7, 6, 0, 3, 2] : (tensor<6x3x2x5x4x4xf32>) -> tensor<5x1x4x4x6x1x2x3xf32>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<5x1x4x4x6x1x2x3xf32>) -> tensor<5x4x4x12x3xf32>
// CHECK-NEXT:     %2 = stablehlo.transpose %1, dims = [1, 2, 0, 3, 4] : (tensor<5x4x4x12x3xf32>) -> tensor<4x4x5x12x3xf32>
// CHECK-NEXT:     %3 = stablehlo.reshape %2 : (tensor<4x4x5x12x3xf32>) -> tensor<4x4x60x3xf32
// CHECK-NEXT{LITERAL}:     %4 = stablehlo.convolution(%3, %cst) dim_numbers = [0, 1, f, b]x[0, 1, i, o]->[0, 1, f, b], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 30 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<4x4x60x3xf32>, tensor<3x3x2x150xf32>) -> tensor<2x2x150x3xf32>
// CHECK-NEXT:     %5 = stablehlo.reshape %4 : (tensor<2x2x150x3xf32>) -> tensor<2x2x5x30x3xf32>
// CHECK-NEXT:     %6 = stablehlo.transpose %5, dims = [2, 0, 1, 3, 4] : (tensor<2x2x5x30x3xf32>) -> tensor<5x2x2x30x3xf32>
// CHECK-NEXT:     %7 = stablehlo.reshape %6 : (tensor<5x2x2x30x3xf32>) -> tensor<5x2x2x6x5x3xf32>
// CHECK-NEXT:     %8 = stablehlo.transpose %7, dims = [3, 5, 4, 0, 2, 1] : (tensor<5x2x2x6x5x3xf32>) -> tensor<6x3x5x5x2x2xf32>
// CHECK-NEXT:     return %8 : tensor<6x3x5x5x2x2xf32>
// CHECK-NEXT:   }
