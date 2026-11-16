// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=slice outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=slice outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

// The limit index of the strided slice is deliberately non-tight: [1:10:3] of a
// 10-element dimension yields 3 elements (indices 1, 4, 7), exactly like the
// tight [1:8:3]. The reverse rule must pad the adjoint back to the full operand
// shape regardless of how the limit was written.
func.func @slice(%x : tensor<4x10xf32>) -> tensor<2x3xf32> {
  %y = stablehlo.slice %x [1:3, 1:10:3] : (tensor<4x10xf32>) -> tensor<2x3xf32>
  func.return %y : tensor<2x3xf32>
}

// FORWARD:  func.func @slice(%arg0: tensor<4x10xf32>, %arg1: tensor<4x10xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.slice %arg1 [1:3, 1:10:3] : (tensor<4x10xf32>) -> tensor<2x3xf32>
// FORWARD-NEXT:    %1 = stablehlo.slice %arg0 [1:3, 1:10:3] : (tensor<4x10xf32>) -> tensor<2x3xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<2x3xf32>, tensor<2x3xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @slice(%arg0: tensor<4x10xf32>, %arg1: tensor<2x3xf32>) -> tensor<4x10xf32> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// REVERSE-NEXT:    %0 = stablehlo.pad %arg1, %cst, low = [1, 1], high = [1, 2], interior = [0, 2] : (tensor<2x3xf32>, tensor<f32>) -> tensor<4x10xf32>
// REVERSE-NEXT:    return %0 : tensor<4x10xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %input = stablehlo.constant dense<[
    [ 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0],
    [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
    [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
    [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0]]> : tensor<4x10xf32>
  %output = stablehlo.constant dense<[[11.0, 14.0, 17.0], [21.0, 24.0, 27.0]]> : tensor<2x3xf32>

  %dinput = stablehlo.constant dense<[
    [ 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0],
    [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
    [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
    [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0]]> : tensor<4x10xf32>
  %fwd_expected = stablehlo.constant dense<[[11.0, 14.0, 17.0], [21.0, 24.0, 27.0]]> : tensor<2x3xf32>

  %fwd:2 = enzyme.fwddiff @slice(%input, %dinput) <{
    activity=[#enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<4x10xf32>, tensor<4x10xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>)

  check.expect_almost_eq %fwd#0, %output : tensor<2x3xf32>
  check.expect_almost_eq %fwd#1, %fwd_expected : tensor<2x3xf32>

  %doutput = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %rev_expected = stablehlo.constant dense<[
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0],
    [0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]> : tensor<4x10xf32>

  %rev:2 = enzyme.autodiff @slice(%input, %doutput) <{
    activity=[#enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<4x10xf32>, tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<4x10xf32>)

  check.expect_almost_eq %rev#0, %output : tensor<2x3xf32>
  check.expect_almost_eq %rev#1, %rev_expected : tensor<4x10xf32>

  func.return
}
