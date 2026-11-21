// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | stablehlo-translate - -allow-unregistered-dialect --interpret

module {
  // 1. #1646: Reverse of non-splat constant folds via ReverseConstProp.
  // CHECK-LABEL: func.func private @reverse_const_prop_fn
  // CHECK-NOT: stablehlo.reverse
  // CHECK: stablehlo.constant dense<{{\[\[}}4.0{{.*}}, 3.0{{.*}}{{\]}}, {{\[}}2.0{{.*}}, 1.0{{.*}}{{\]\]}}>
  func.func private @reverse_const_prop_fn() -> tensor<2x2xf32> {
    %cst = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %r = stablehlo.reverse %cst, dims = [0, 1] : tensor<2x2xf32>
    return %r : tensor<2x2xf32>
  }

  // 2. #1648: Reverse(reverse(x, dims1), dims2) folds to symmetric difference (or noop).
  // CHECK-LABEL: func.func private @reverse_reverse_cancel
  // CHECK-NOT: stablehlo.reverse
  // CHECK: return %arg0 : tensor<3x4xf32>
  func.func private @reverse_reverse_cancel(%x: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %r1 = stablehlo.reverse %x, dims = [0, 1] : tensor<3x4xf32>
    %r2 = stablehlo.reverse %r1, dims = [0, 1] : tensor<3x4xf32>
    return %r2 : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func private @reverse_reverse_symdiff
  // CHECK: %[[R:.*]] = stablehlo.reverse %arg0, dims = [1] : tensor<3x4xf32>
  // CHECK-NOT: stablehlo.reverse
  // CHECK: return %[[R]] : tensor<3x4xf32>
  func.func private @reverse_reverse_symdiff(%x: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %r1 = stablehlo.reverse %x, dims = [0, 1] : tensor<3x4xf32>
    %r2 = stablehlo.reverse %r1, dims = [0] : tensor<3x4xf32>
    return %r2 : tensor<3x4xf32>
  }

  // 3. #1647: Slice of reverse pushes slice before reverse (including strided multi-element slices).
  // CHECK-LABEL: func.func private @slice_of_reverse
  // CHECK: %[[S:.*]] = stablehlo.slice %arg0 [1:3, 1:6:2] : (tensor<4x6xf32>) -> tensor<2x3xf32>
  // CHECK-NEXT: %[[R:.*]] = stablehlo.reverse %[[S]], dims = [0, 1] : tensor<2x3xf32>
  // CHECK-NEXT: return %[[R]] : tensor<2x3xf32>
  func.func private @slice_of_reverse(%x: tensor<4x6xf32>) -> tensor<2x3xf32> {
    %r = stablehlo.reverse %x, dims = [0, 1] : tensor<4x6xf32>
    %s = stablehlo.slice %r [1:3, 0:5:2] : (tensor<4x6xf32>) -> tensor<2x3xf32>
    return %s : tensor<2x3xf32>
  }

  func.func @main() {
    %c = func.call @reverse_const_prop_fn() : () -> tensor<2x2xf32>
    check.expect_almost_eq_const %c, dense<[[4.0, 3.0], [2.0, 1.0]]> : tensor<2x2xf32>

    %x = stablehlo.constant dense<[
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0]
    ]> : tensor<3x4xf32>
    %rr0 = func.call @reverse_reverse_cancel(%x) : (tensor<3x4xf32>) -> tensor<3x4xf32>
    check.expect_almost_eq_const %rr0, dense<[
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0]
    ]> : tensor<3x4xf32>

    %rr1 = func.call @reverse_reverse_symdiff(%x) : (tensor<3x4xf32>) -> tensor<3x4xf32>
    check.expect_almost_eq_const %rr1, dense<[
      [4.0, 3.0, 2.0, 1.0],
      [8.0, 7.0, 6.0, 5.0],
      [12.0, 11.0, 10.0, 9.0]
    ]> : tensor<3x4xf32>

    %y = stablehlo.constant dense<[
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
      [31.0, 32.0, 33.0, 34.0, 35.0, 36.0]
    ]> : tensor<4x6xf32>
    %sr = func.call @slice_of_reverse(%y) : (tensor<4x6xf32>) -> tensor<2x3xf32>
    // reverse(y, [0,1]) has row 1 = [26,25,24,23,22,21], row 2 = [16,15,14,13,12,11]
    // [0:5:2] along col picks indices 0, 2, 4 -> [26, 24, 22] and [16, 14, 12].
    check.expect_almost_eq_const %sr, dense<[
      [26.0, 24.0, 22.0],
      [16.0, 14.0, 12.0]
    ]> : tensor<2x3xf32>
    return
  }
}
