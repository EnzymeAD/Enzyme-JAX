// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=transpose outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=transpose outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

func.func @transpose(%input : tensor<2x3xf32>) -> tensor<3x2xf32> {
  %y = stablehlo.transpose %input, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  func.return %y : tensor<3x2xf32>
}

// FORWARD:  func.func @transpose(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> (tensor<3x2xf32>, tensor<3x2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
// FORWARD-NEXT:    %1 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<3x2xf32>, tensor<3x2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @transpose(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x3xf32> {
// REVERSE-NEXT:    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
// REVERSE-NEXT:    return %0 : tensor<2x3xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %input = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %output = stablehlo.constant dense<[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]> : tensor<3x2xf32>

  %expected_fwd = stablehlo.constant dense<[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]> : tensor<3x2xf32>
  %dinput_fwd = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>

  %fwd:2 = enzyme.fwddiff @transpose(%input, %dinput_fwd) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<3x2xf32>, tensor<3x2xf32>)

  check.expect_almost_eq %fwd#0, %output : tensor<3x2xf32>
  check.expect_almost_eq %fwd#1, %expected_fwd : tensor<3x2xf32>

  %expected_rev = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %dinput_rev = stablehlo.constant dense<[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]> : tensor<3x2xf32>

  %rev:2 = enzyme.autodiff @transpose(%input, %dinput_rev) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2x3xf32>, tensor<3x2xf32>) -> (tensor<3x2xf32>, tensor<2x3xf32>)

  check.expect_almost_eq %rev#0, %output : tensor<3x2xf32>
  check.expect_almost_eq %rev#1, %expected_rev : tensor<2x3xf32>

  func.return
}
