// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reverse outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reverse outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

func.func @reverse(%x : tensor<2x3xf32>) -> tensor<2x3xf32> {
  %y = stablehlo.reverse %x, dims = [1] : (tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %y : tensor<2x3xf32>
}

// FORWARD:  func.func @reverse(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.reverse %arg1, dims = [1] : tensor<2x3xf32>
// FORWARD-NEXT:    %1 = stablehlo.reverse %arg0, dims = [1] : tensor<2x3xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<2x3xf32>, tensor<2x3xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @reverse(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
// REVERSE-NEXT:    %0 = stablehlo.reverse %arg1, dims = [1] : tensor<2x3xf32>
// REVERSE-NEXT:    return %0 : tensor<2x3xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %input = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %output = stablehlo.constant dense<[[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]> : tensor<2x3xf32>

  %dinput = stablehlo.constant dense<[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]> : tensor<2x3xf32>
  %expected = stablehlo.constant dense<[[9.0, 8.0, 7.0], [12.0, 11.0, 10.0]]> : tensor<2x3xf32>

  %fwd:2 = enzyme.fwddiff @reverse(%input, %dinput) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>)

  check.expect_almost_eq %fwd#0, %output : tensor<2x3xf32>
  check.expect_almost_eq %fwd#1, %expected : tensor<2x3xf32>

  %rev:2 = enzyme.autodiff @reverse(%input, %dinput) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>)

  check.expect_almost_eq %rev#0, %output : tensor<2x3xf32>
  check.expect_almost_eq %rev#1, %expected : tensor<2x3xf32>

  func.return
}
