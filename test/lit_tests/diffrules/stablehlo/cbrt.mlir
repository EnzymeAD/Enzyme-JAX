// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=cbrt outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=cbrt outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @cbrt(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = stablehlo.cbrt %x : (tensor<2xf32>) -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:  func.func @cbrt(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %0 = stablehlo.cbrt %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %0, %0 : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.multiply %cst, %1 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.divide %arg1, %2 : tensor<2xf32>
// FORWARD-NEXT:    return %0, %3 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @cbrt(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = stablehlo.cbrt %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.multiply %0, %0 : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.multiply %cst, %1 : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.divide %arg1, %2 : tensor<2xf32>
// REVERSE-NEXT:    return %3 : tensor<2xf32>
// REVERSE-NEXT:  }

// Exercise both signs. cbrt is real and smooth for x < 0 (cbrt(-8) = -2), so the
// true gradient 1/(3*cbrt(x)^2) is a finite real with the same magnitude as on
// the positives. The old pow(x,-2/3) rule returned NaN on negatives. See issue #2571.
func.func @main() {
  %dx = stablehlo.constant dense<1.0> : tensor<2xf32>

  // Positive inputs: unchanged from the pre-fix behavior.
  %xp = stablehlo.constant dense<[8.0, 27.0]> : tensor<2xf32>
  %outp = stablehlo.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %expectedp = stablehlo.constant dense<[0.083333336, 0.037037037]> : tensor<2xf32>

  %fwdp:2 = enzyme.fwddiff @cbrt(%xp, %dx) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwdp#0, %outp : tensor<2xf32>
  check.expect_almost_eq %fwdp#1, %expectedp : tensor<2xf32>

  %revp:2 = enzyme.autodiff @cbrt(%xp, %dx) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %revp#0, %outp : tensor<2xf32>
  check.expect_almost_eq %revp#1, %expectedp : tensor<2xf32>

  // Negative inputs: the regression this PR fixes. Same gradient magnitude.
  %xn = stablehlo.constant dense<[-8.0, -27.0]> : tensor<2xf32>
  %outn = stablehlo.constant dense<[-2.0, -3.0]> : tensor<2xf32>
  %expectedn = stablehlo.constant dense<[0.083333336, 0.037037037]> : tensor<2xf32>

  %fwdn:2 = enzyme.fwddiff @cbrt(%xn, %dx) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwdn#0, %outn : tensor<2xf32>
  check.expect_almost_eq %fwdn#1, %expectedn : tensor<2xf32>

  %revn:2 = enzyme.autodiff @cbrt(%xn, %dx) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %revn#0, %outn : tensor<2xf32>
  check.expect_almost_eq %revn#1, %expectedn : tensor<2xf32>

  func.return
}
