// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=log outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt  | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=log outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

func.func @log(%x : tensor<3xf32>) -> tensor<3xf32> {
  %y = stablehlo.log %x : tensor<3xf32>
  func.return %y : tensor<3xf32>
}

// FORWARD:  func.func @log(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.divide %arg1, %arg0 : tensor<3xf32>
// FORWARD-NEXT:    %1 = stablehlo.log %arg0 : tensor<3xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<3xf32>, tensor<3xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @log(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
// REVERSE-NEXT:    %0 = stablehlo.divide %arg1, %arg0 : tensor<3xf32>
// REVERSE-NEXT:    return %0 : tensor<3xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %input = stablehlo.constant dense<[0.1, 1.0, 10.0]> : tensor<3xf32>
  %output = stablehlo.constant dense<[-2.3025850929940455, 0.0, 2.3025850929940455]> : tensor<3xf32>
  %expected = stablehlo.constant dense<[10.0, 1.0, 0.1]> : tensor<3xf32>

  %dinput = stablehlo.constant dense<1.0> : tensor<3xf32>

  // fwd diff
  %fwd_res:2 = enzyme.fwddiff @log(%input, %dinput) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)

  check.expect_almost_eq %fwd_res#0, %output : tensor<3xf32>
  check.expect_almost_eq %fwd_res#1, %expected : tensor<3xf32>

  // rev diff
  %rev_res:2 = enzyme.autodiff @log(%input, %dinput) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)

  check.expect_almost_eq %rev_res#0, %output : tensor<3xf32>
  check.expect_almost_eq %rev_res#1, %expected : tensor<3xf32>

  func.return
}
