// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=power outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=power outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @power(%a : tensor<5xf32>, %b : tensor<5xf32>) -> tensor<5xf32> {
  %c = stablehlo.power %a, %b : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  func.return %c : tensor<5xf32>
}

// FORWARD:  func.func @power(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>, %arg2: tensor<5xf32>, %arg3: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
// FORWARD-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<5xf32>
// FORWARD-NEXT:    %0 = stablehlo.subtract %arg2, %cst : tensor<5xf32>
// FORWARD-NEXT:    %1 = stablehlo.power %arg0, %0 : tensor<5xf32>
// FORWARD-NEXT:    %2 = stablehlo.multiply %arg2, %1 : tensor<5xf32>
// FORWARD-NEXT:    %3 = stablehlo.multiply %arg1, %2 : tensor<5xf32>
// FORWARD-NEXT:    %4 = stablehlo.power %arg0, %arg2 : tensor<5xf32>
// FORWARD-NEXT:    %5 = stablehlo.log %arg0 : tensor<5xf32>
// FORWARD-NEXT:    %6 = stablehlo.multiply %4, %5 : tensor<5xf32>
// FORWARD-NEXT:    %7 = stablehlo.multiply %arg3, %6 : tensor<5xf32>
// FORWARD-NEXT:    %8 = stablehlo.add %3, %7 : tensor<5xf32>
// FORWARD-NEXT:    return %4, %8 : tensor<5xf32>, tensor<5xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @power(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>, %arg2: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<5xf32>
// REVERSE-NEXT:    %0 = stablehlo.subtract %arg1, %cst : tensor<5xf32>
// REVERSE-NEXT:    %1 = stablehlo.power %arg0, %0 : tensor<5xf32>
// REVERSE-NEXT:    %2 = stablehlo.multiply %arg1, %1 : tensor<5xf32>
// REVERSE-NEXT:    %3 = stablehlo.multiply %arg2, %2 : tensor<5xf32>
// REVERSE-NEXT:    %4 = stablehlo.power %arg0, %arg1 : tensor<5xf32>
// REVERSE-NEXT:    %5 = stablehlo.log %arg0 : tensor<5xf32>
// REVERSE-NEXT:    %6 = stablehlo.multiply %4, %5 : tensor<5xf32>
// REVERSE-NEXT:    %7 = stablehlo.multiply %arg2, %6 : tensor<5xf32>
// REVERSE-NEXT:    return %3, %7 : tensor<5xf32>, tensor<5xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %a = stablehlo.constant dense<[2.0, 3.0, 4.0, 6.0, 10.0]> : tensor<5xf32>
  %b = stablehlo.constant dense<[2.0, 1.0, 0.5, 0.0, -1.0]> : tensor<5xf32>
  %output = stablehlo.constant dense<[4.0, 3.0, 2.0, 1.0, 0.1]> : tensor<5xf32>

  %expected_da = stablehlo.constant dense<[4.0, 1.0, 0.25, 0.0, -0.01]> : tensor<5xf32>
  %expected_db = stablehlo.constant dense<[2.772588722239978, 3.2958368660043305, 2.7725887222398486, 1.791759469228088, 0.23025850929943448]> : tensor<5xf32>

  %done = stablehlo.constant dense<1.0> : tensor<5xf32>
  %dzero = stablehlo.constant dense<0.0> : tensor<5xf32>

  %fwd_a:2 = enzyme.fwddiff @power(%a, %done, %b, %dzero) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<5xf32>, tensor<5xf32>, tensor<5xf32>, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)

  check.expect_almost_eq %fwd_a#0, %output : tensor<5xf32>
  check.expect_almost_eq %fwd_a#1, %expected_da : tensor<5xf32>

  %fwd_b:2 = enzyme.fwddiff @power(%a, %dzero, %b, %done) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<5xf32>, tensor<5xf32>, tensor<5xf32>, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)

  check.expect_almost_eq %fwd_b#0, %output : tensor<5xf32>
  check.expect_almost_eq %fwd_b#1, %expected_db : tensor<5xf32>

  %rev:3 = enzyme.autodiff @power(%a, %b, %done) {
    activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<5xf32>, tensor<5xf32>, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>, tensor<5xf32>)

  check.expect_almost_eq %rev#0, %output : tensor<5xf32>
  check.expect_almost_eq %rev#1, %expected_da : tensor<5xf32>
  check.expect_almost_eq %rev#2, %expected_db : tensor<5xf32>

  func.return
}
