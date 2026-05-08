// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=divide outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=divide outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

func.func @divide(%a : tensor<2xf32>, %b : tensor<2xf32>) -> tensor<2xf32> {
  %c = stablehlo.divide %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %c : tensor<2xf32>
}

// FORWARD:  func.func @divide(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.multiply %arg1, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %arg3, %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.subtract %0, %1 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.multiply %arg2, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    %4 = stablehlo.divide %2, %3 : tensor<2xf32>
// FORWARD-NEXT:    %5 = stablehlo.divide %arg0, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    return %5, %4 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @divide(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// REVERSE-NEXT:    %0 = stablehlo.divide %arg2, %arg1 : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.divide %arg0, %arg1 : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.multiply %0, %1 : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.negate %2 : tensor<2xf32>
// REVERSE-NEXT:    return %0, %3 : tensor<2xf32>, tensor<2xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %a = stablehlo.constant dense<[4.0, 2.0]> : tensor<2xf32>
  %b = stablehlo.constant dense<[2.0, 5.0]> : tensor<2xf32>
  %output = stablehlo.constant dense<[2.0, 0.4]> : tensor<2xf32>

  %expected_da = stablehlo.constant dense<[0.5, 0.2]> : tensor<2xf32>
  %expected_db = stablehlo.constant dense<[-1.0, -0.08]> : tensor<2xf32>

  %dzero = stablehlo.constant dense<0.0> : tensor<2xf32>
  %done = stablehlo.constant dense<1.0> : tensor<2xf32>

  // fwd diff wrt a
  %fwd_a:2 = enzyme.fwddiff @divide(%a, %done, %b, %dzero) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd_a#0, %output : tensor<2xf32>
  check.expect_almost_eq %fwd_a#1, %expected_da : tensor<2xf32>

  // fwd diff wrt b
  %fwd_b:2 = enzyme.fwddiff @divide(%a, %dzero, %b, %done) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd_b#0, %output : tensor<2xf32>
  check.expect_almost_eq %fwd_b#1, %expected_db : tensor<2xf32>

  // rev diff
  %rev:3 = enzyme.autodiff @divide(%a, %b, %done) {
    activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %rev#0, %output : tensor<2xf32>
  check.expect_almost_eq %rev#1, %expected_da : tensor<2xf32>
  check.expect_almost_eq %rev#2, %expected_db : tensor<2xf32>

  func.return
}
