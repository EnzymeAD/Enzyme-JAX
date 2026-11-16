// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=log2 outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt  | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=log2 outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE
// The interpreter run executes @main below: its check.expect_almost_eq ops
// compare the primal and the forward- and reverse-mode derivatives that enzyme
// produced against the expected values, and a mismatch makes
// stablehlo-translate exit non-zero.
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

func.func @log2(%x : tensor<3xf32>) -> tensor<3xf32> {
  %y = stablehlo.log2 %x : tensor<3xf32>
  func.return %y : tensor<3xf32>
}

// FORWARD:  func.func @log2(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
// FORWARD-NEXT:    %[[LN2:.+]] = stablehlo.constant dense<0.693147182> : tensor<3xf32>
// FORWARD-NEXT:    %[[S:.+]] = stablehlo.multiply %arg0, %[[LN2]] : tensor<3xf32>
// FORWARD-NEXT:    %[[D:.+]] = stablehlo.divide %arg1, %[[S]] : tensor<3xf32>
// FORWARD-NEXT:    %[[L:.+]] = stablehlo.log2 %arg0 : tensor<3xf32>
// FORWARD-NEXT:    return %[[L]], %[[D]] : tensor<3xf32>, tensor<3xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @log2(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
// REVERSE-NEXT:    %[[LN2:.+]] = stablehlo.constant dense<0.693147182> : tensor<3xf32>
// REVERSE-NEXT:    %[[S:.+]] = stablehlo.multiply %arg0, %[[LN2]] : tensor<3xf32>
// REVERSE-NEXT:    %[[D:.+]] = stablehlo.divide %arg1, %[[S]] : tensor<3xf32>
// REVERSE-NEXT:    return %[[D]] : tensor<3xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %input = stablehlo.constant dense<[0.5, 1.0, 8.0]> : tensor<3xf32>
  %output = stablehlo.constant dense<[-1.0, 0.0, 3.0]> : tensor<3xf32>
  // 1 / (x * ln(2))
  %expected = stablehlo.constant dense<[2.8853900817779268, 1.4426950408889634, 0.18033688011112042]> : tensor<3xf32>

  %dinput = stablehlo.constant dense<1.0> : tensor<3xf32>

  // fwd diff
  %fwd_res:2 = enzyme.fwddiff @log2(%input, %dinput) <{
    activity=[#enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)

  check.expect_almost_eq %fwd_res#0, %output : tensor<3xf32>
  check.expect_almost_eq %fwd_res#1, %expected : tensor<3xf32>

  // rev diff
  %rev_res:2 = enzyme.autodiff @log2(%input, %dinput) <{
    activity=[#enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)

  check.expect_almost_eq %rev_res#0, %output : tensor<3xf32>
  check.expect_almost_eq %rev_res#1, %expected : tensor<3xf32>

  func.return
}
