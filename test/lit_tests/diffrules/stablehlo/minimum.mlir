// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=minimum outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=minimum outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

func.func @minimum(%a : tensor<4xf32>, %b : tensor<4xf32>) -> tensor<4xf32> {
  %c = stablehlo.minimum %a, %b : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %c : tensor<4xf32>
}

// FORWARD:  func.func @minimum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.compare  LT, %arg2, %arg0 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// FORWARD-NEXT:    %1 = stablehlo.select %0, %arg3, %arg1 : tensor<4xi1>, tensor<4xf32>
// FORWARD-NEXT:    %2 = stablehlo.minimum %arg0, %arg2 : tensor<4xf32>
// FORWARD-NEXT:    return %2, %1 : tensor<4xf32>, tensor<4xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @minimum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// REVERSE-NEXT:    %0 = stablehlo.compare  LT, %arg1, %arg0 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// REVERSE-NEXT:    %1 = stablehlo.select %0, %cst, %arg2 : tensor<4xi1>, tensor<4xf32>
// REVERSE-NEXT:    %2 = stablehlo.select %0, %arg2, %cst : tensor<4xi1>, tensor<4xf32>
// REVERSE-NEXT:    return %1, %2 : tensor<4xf32>, tensor<4xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %a = stablehlo.constant dense<[1.0, -2.0, 1.5, 100.0]> : tensor<4xf32>
  %b = stablehlo.constant dense<[2.0, -1.0, 1.5, 50.0]> : tensor<4xf32>
  %output = stablehlo.constant dense<[1.0, -2.0, 1.5, 50.0]> : tensor<4xf32>

  %expected_da = stablehlo.constant dense<[1.0, 1.0, 1.0, 0.0]> : tensor<4xf32>
  %expected_db = stablehlo.constant dense<[0.0, 0.0, 0.0, 1.0]> : tensor<4xf32>

  %dzero = stablehlo.constant dense<0.0> : tensor<4xf32>
  %done = stablehlo.constant dense<1.0> : tensor<4xf32>

  // fwd diff
  %fwd_res_a:2 = enzyme.fwddiff @minimum(%a, %done, %b, %dzero) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)

  check.expect_almost_eq %fwd_res_a#0, %output : tensor<4xf32>
  check.expect_almost_eq %fwd_res_a#1, %expected_da : tensor<4xf32>

  %fwd_res_b:2 = enzyme.fwddiff @minimum(%a, %dzero, %b, %done) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)

  check.expect_almost_eq %fwd_res_b#0, %output : tensor<4xf32>
  check.expect_almost_eq %fwd_res_b#1, %expected_db : tensor<4xf32>

  // rev diff
  %rev_res:3 = enzyme.autodiff @minimum(%a, %b, %done) {
    activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)

  check.expect_almost_eq %rev_res#0, %output : tensor<4xf32>
  check.expect_almost_eq %rev_res#1, %expected_da : tensor<4xf32>
  check.expect_almost_eq %rev_res#2, %expected_db : tensor<4xf32>

  func.return
}
