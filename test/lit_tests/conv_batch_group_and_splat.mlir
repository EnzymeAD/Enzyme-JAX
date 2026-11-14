// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | stablehlo-translate - -allow-unregistered-dialect --interpret

module {
  // 1. #1572: Repeated non-splat kernel along `o` drops `batch_group_count` to 1.
  // CHECK-LABEL: func.func private @conv_batch_group_repeated_kernel
  // CHECK-NOT: batch_group_count = 4
  // CHECK: stablehlo.convolution{{.*}}batch_group_count = 1 : i64
  func.func private @conv_batch_group_repeated_kernel(
      %lhs: tensor<4x2x8xf32>, %ker: tensor<3x2x5xf32>) -> tensor<2x20x2xf32> {
    %bcast = stablehlo.broadcast_in_dim %ker, dims = [0, 1, 3] : (tensor<3x2x5xf32>) -> tensor<3x2x4x5xf32>
    %rep_ker = stablehlo.reshape %bcast : (tensor<3x2x4x5xf32>) -> tensor<3x2x20xf32>
    %conv = stablehlo.convolution(%lhs, %rep_ker)
        dim_numbers = [0, f, b]x[0, i, o]->[0, f, b],
        window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]}
        {batch_group_count = 4 : i64, feature_group_count = 1 : i64}
        : (tensor<4x2x8xf32>, tensor<3x2x20xf32>) -> tensor<2x20x2xf32>
    return %conv : tensor<2x20x2xf32>
  }

  // 2. #1572 with [b, f, 0] output layout (`bOut < fOut`).
  // CHECK-LABEL: func.func private @conv_batch_group_bf0_layout
  // CHECK-NOT: batch_group_count = 2
  // CHECK: stablehlo.convolution{{.*}}batch_group_count = 1 : i64
  func.func private @conv_batch_group_bf0_layout(
      %lhs: tensor<6x2x4xf32>, %ker: tensor<2x3x3xf32>) -> tensor<3x6x2xf32> {
    %bcast = stablehlo.broadcast_in_dim %ker, dims = [0, 2, 3] : (tensor<2x3x3xf32>) -> tensor<2x2x3x3xf32>
    %rep_ker = stablehlo.reshape %bcast : (tensor<2x2x3x3xf32>) -> tensor<2x6x3xf32>
    %conv = stablehlo.convolution(%lhs, %rep_ker)
        dim_numbers = [b, f, 0]x[i, o, 0]->[b, f, 0],
        window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]}
        {batch_group_count = 2 : i64, feature_group_count = 1 : i64}
        : (tensor<6x2x4xf32>, tensor<2x6x3xf32>) -> tensor<3x6x2xf32>
    return %conv : tensor<3x6x2xf32>
  }

  // 3. #1474: Unpadded splat LHS folds away `stablehlo.convolution` completely to reduce + mul + broadcast.
  // CHECK-LABEL: func.func private @conv_splat_lhs_unpadded
  // CHECK-NOT: stablehlo.convolution
  // CHECK: stablehlo.reduce
  // CHECK: stablehlo.broadcast_in_dim
  func.func private @conv_splat_lhs_unpadded(%rhs: tensor<3x2x4xf32>) -> tensor<2x3x5xf32> {
    %lhs = stablehlo.constant dense<2.5> : tensor<2x2x8xf32>
    %conv = stablehlo.convolution(%lhs, %rhs)
        dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0],
        window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]}
        {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
        : (tensor<2x2x8xf32>, tensor<3x2x4xf32>) -> tensor<2x3x5xf32>
    return %conv : tensor<2x3x5xf32>
  }

  // 4. #1474: Padded strided splat LHS collapses batch, reduces `i`, and shrinks interior spatial dim.
  // CHECK-LABEL: func.func private @conv_splat_lhs_padded_interior
  // CHECK: stablehlo.constant dense<2.000000e+00> : tensor<1x1x4xf32>
  // CHECK: stablehlo.convolution{{.*}} : (tensor<1x1x4xf32>, tensor<1x1x4xf32>) -> tensor<1x1x5xf32>
  func.func private @conv_splat_lhs_padded_interior(%rhs: tensor<1x2x4xf32>) -> tensor<2x1x9xf32> {
    %lhs = stablehlo.constant dense<2.0> : tensor<2x2x12xf32>
    %conv = stablehlo.convolution(%lhs, %rhs)
        dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0],
        window = {stride = [2], pad = [[4, 4]], rhs_dilate = [1]}
        {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
        : (tensor<2x2x12xf32>, tensor<1x2x4xf32>) -> tensor<2x1x9xf32>
    return %conv : tensor<2x1x9xf32>
  }

  // 5. #1474: Padded splat RHS collapses `f` (`i`) and `o` (`O -> 1` broadcasted).
  // CHECK-LABEL: func.func private @conv_splat_rhs_padded
  // CHECK: stablehlo.reduce
  // CHECK: stablehlo.convolution{{.*}} : (tensor<2x1x6xf32>, tensor<1x1x3xf32>) -> tensor<2x1x6xf32>
  func.func private @conv_splat_rhs_padded(%lhs: tensor<2x3x6xf32>) -> tensor<2x4x6xf32> {
    %rhs = stablehlo.constant dense<3.0> : tensor<3x4x3xf32>
    %conv = stablehlo.convolution(%lhs, %rhs)
        dim_numbers = [b, f, 0]x[i, o, 0]->[b, f, 0],
        window = {stride = [1], pad = [[1, 1]], rhs_dilate = [1]}
        {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
        : (tensor<2x3x6xf32>, tensor<3x4x3xf32>) -> tensor<2x4x6xf32>
    return %conv : tensor<2x4x6xf32>
  }

  func.func @main() {
    // Verify Test 1 against explicit per-group regular convolutions
    %lhs1_flat = stablehlo.iota dim = 0 : tensor<64xf32>
    %lhs1 = stablehlo.reshape %lhs1_flat : (tensor<64xf32>) -> tensor<4x2x8xf32>
    %ker1_flat = stablehlo.iota dim = 0 : tensor<30xf32>
    %ker1 = stablehlo.reshape %ker1_flat : (tensor<30xf32>) -> tensor<3x2x5xf32>
    %got1 = func.call @conv_batch_group_repeated_kernel(%lhs1, %ker1) : (tensor<4x2x8xf32>, tensor<3x2x5xf32>) -> tensor<2x20x2xf32>

    %l_g0 = stablehlo.slice %lhs1 [0:4, 0:2, 0:2] : (tensor<4x2x8xf32>) -> tensor<4x2x2xf32>
    %l_g1 = stablehlo.slice %lhs1 [0:4, 0:2, 2:4] : (tensor<4x2x8xf32>) -> tensor<4x2x2xf32>
    %l_g2 = stablehlo.slice %lhs1 [0:4, 0:2, 4:6] : (tensor<4x2x8xf32>) -> tensor<4x2x2xf32>
    %l_g3 = stablehlo.slice %lhs1 [0:4, 0:2, 6:8] : (tensor<4x2x8xf32>) -> tensor<4x2x2xf32>
    %c_g0 = stablehlo.convolution(%l_g0, %ker1) dim_numbers = [0, f, b]x[0, i, o]->[0, f, b], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<4x2x2xf32>, tensor<3x2x5xf32>) -> tensor<2x5x2xf32>
    %c_g1 = stablehlo.convolution(%l_g1, %ker1) dim_numbers = [0, f, b]x[0, i, o]->[0, f, b], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<4x2x2xf32>, tensor<3x2x5xf32>) -> tensor<2x5x2xf32>
    %c_g2 = stablehlo.convolution(%l_g2, %ker1) dim_numbers = [0, f, b]x[0, i, o]->[0, f, b], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<4x2x2xf32>, tensor<3x2x5xf32>) -> tensor<2x5x2xf32>
    %c_g3 = stablehlo.convolution(%l_g3, %ker1) dim_numbers = [0, f, b]x[0, i, o]->[0, f, b], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<4x2x2xf32>, tensor<3x2x5xf32>) -> tensor<2x5x2xf32>
    %exp1 = stablehlo.concatenate %c_g0, %c_g1, %c_g2, %c_g3, dim = 1 : (tensor<2x5x2xf32>, tensor<2x5x2xf32>, tensor<2x5x2xf32>, tensor<2x5x2xf32>) -> tensor<2x20x2xf32>
    check.expect_almost_eq %got1, %exp1 : tensor<2x20x2xf32>

    // Verify Test 2 (`[b, f, 0]` layout)
    %lhs2_flat = stablehlo.iota dim = 0 : tensor<48xf32>
    %lhs2 = stablehlo.reshape %lhs2_flat : (tensor<48xf32>) -> tensor<6x2x4xf32>
    %ker2_flat = stablehlo.iota dim = 0 : tensor<18xf32>
    %ker2 = stablehlo.reshape %ker2_flat : (tensor<18xf32>) -> tensor<2x3x3xf32>
    %got2 = func.call @conv_batch_group_bf0_layout(%lhs2, %ker2) : (tensor<6x2x4xf32>, tensor<2x3x3xf32>) -> tensor<3x6x2xf32>
    %l2_g0 = stablehlo.slice %lhs2 [0:3, 0:2, 0:4] : (tensor<6x2x4xf32>) -> tensor<3x2x4xf32>
    %l2_g1 = stablehlo.slice %lhs2 [3:6, 0:2, 0:4] : (tensor<6x2x4xf32>) -> tensor<3x2x4xf32>
    %c2_g0 = stablehlo.convolution(%l2_g0, %ker2) dim_numbers = [b, f, 0]x[i, o, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x2x4xf32>, tensor<2x3x3xf32>) -> tensor<3x3x2xf32>
    %c2_g1 = stablehlo.convolution(%l2_g1, %ker2) dim_numbers = [b, f, 0]x[i, o, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x2x4xf32>, tensor<2x3x3xf32>) -> tensor<3x3x2xf32>
    %exp2 = stablehlo.concatenate %c2_g0, %c2_g1, dim = 1 : (tensor<3x3x2xf32>, tensor<3x3x2xf32>) -> tensor<3x6x2xf32>
    check.expect_almost_eq %got2, %exp2 : tensor<3x6x2xf32>

    // Verify Test 3 (unpadded splat LHS)
    %rhs3 = stablehlo.constant dense<[
      [[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]],
      [[-1.0, 0.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
      [[0.0, 2.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0]]
    ]> : tensor<3x2x4xf32>
    // Channel sums: o=0 -> 12.0 * 2.5 = 30.0; o=1 -> 6.0 * 2.5 = 15.0; o=2 -> 4.0 * 2.5 = 10.0
    %got3 = func.call @conv_splat_lhs_unpadded(%rhs3) : (tensor<3x2x4xf32>) -> tensor<2x3x5xf32>
    check.expect_almost_eq_const %got3, dense<[
      [[30.0, 30.0, 30.0, 30.0, 30.0], [15.0, 15.0, 15.0, 15.0, 15.0], [10.0, 10.0, 10.0, 10.0, 10.0]],
      [[30.0, 30.0, 30.0, 30.0, 30.0], [15.0, 15.0, 15.0, 15.0, 15.0], [10.0, 10.0, 10.0, 10.0, 10.0]]
    ]> : tensor<2x3x5xf32>

    // Verify Test 4 (padded strided splat LHS with collapsed interior)
    %rhs4 = stablehlo.constant dense<[[[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]]> : tensor<1x2x4xf32>
    %got4 = func.call @conv_splat_lhs_padded_interior(%rhs4) : (tensor<1x2x4xf32>) -> tensor<2x1x9xf32>
    check.expect_almost_eq_const %got4, dense<[
      [[0.0, 154.0, 220.0, 220.0, 220.0, 220.0, 220.0, 66.0, 0.0]],
      [[0.0, 154.0, 220.0, 220.0, 220.0, 220.0, 220.0, 66.0, 0.0]]
    ]> : tensor<2x1x9xf32>

    // Verify Test 5 (padded splat RHS)
    %lhs5 = stablehlo.constant dense<1.0> : tensor<2x3x6xf32>
    // Wait: make lhs5 non-splat so it exercises Case 2 (splat RHS, non-splat LHS)!
    %lhs5_iota = stablehlo.iota dim = 2 : tensor<2x3x6xf32>
    %got5 = func.call @conv_splat_rhs_padded(%lhs5_iota) : (tensor<2x3x6xf32>) -> tensor<2x4x6xf32>
    // Each of the 3 channels has [0, 1, 2, 3, 4, 5], so channel sum is [0, 3, 6, 9, 12, 15].
    // Padded with [0, ..., 0], 3-element window sums * 3.0:
    // u=0: (0+0+3)*3 = 9; u=1: (0+3+6)*3 = 27; u=2: (3+6+9)*3 = 54;
    // u=3: (6+9+12)*3 = 81; u=4: (9+12+15)*3 = 108; u=5: (12+15+0)*3 = 81.
    check.expect_almost_eq_const %got5, dense<[
      [[9.0, 27.0, 54.0, 81.0, 108.0, 81.0],
       [9.0, 27.0, 54.0, 81.0, 108.0, 81.0],
       [9.0, 27.0, 54.0, 81.0, 108.0, 81.0],
       [9.0, 27.0, 54.0, 81.0, 108.0, 81.0]],
      [[9.0, 27.0, 54.0, 81.0, 108.0, 81.0],
       [9.0, 27.0, 54.0, 81.0, 108.0, 81.0],
       [9.0, 27.0, 54.0, 81.0, 108.0, 81.0],
       [9.0, 27.0, 54.0, 81.0, 108.0, 81.0]]
    ]> : tensor<2x4x6xf32>
    return
  }
}
