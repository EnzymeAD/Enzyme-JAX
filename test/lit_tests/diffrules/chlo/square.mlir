// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=square outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=square outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse --verify-each=0 | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --chlo-legalize-to-stablehlo --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @square(%x : tensor<5xf32>) -> tensor<5xf32> {
  %y = chlo.square %x : tensor<5xf32> -> tensor<5xf32>
  func.return %y : tensor<5xf32>
}

// FORWARD:  func.func @square(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
// FORWARD-NEXT:   %[[CST:.*]] = chlo.constant dense<2.000000e+00> : tensor<5xf32>
// FORWARD-NEXT:   %[[DIFF:.*]] = stablehlo.multiply %arg1, %[[CST]] : tensor<5xf32>
// FORWARD-NEXT:   %[[PRIMAL:.*]] = chlo.square %arg0 : tensor<5xf32>
// FORWARD-NEXT:   return %[[PRIMAL]], %[[DIFF]] : tensor<5xf32>, tensor<5xf32>
// FORWARD-NEXT: }

// REVERSE:  func.func @square(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> {
// REVERSE-NEXT:   %[[CST:.*]] = chlo.constant dense<2.000000e+00> : tensor<5xf32>
// REVERSE-NEXT:   %[[DIFF:.*]] = stablehlo.multiply %arg1, %[[CST]] : tensor<5xf32>
// REVERSE-NEXT:   return %[[DIFF]] : tensor<5xf32>
// REVERSE-NEXT: }

func.func @main() {
  %x = stablehlo.constant dense<[0.0, 1.0, 2.5, -3.0, 4.0]> : tensor<5xf32>
  %output = stablehlo.constant dense<[0.0, 1.0, 6.25, 9.0, 16.0]> : tensor<5xf32>
  %expected = stablehlo.constant dense<[0.0, 2.0, 5.0, -6.0, 8.0]> : tensor<5xf32>

  %d = stablehlo.constant dense<1.0> : tensor<5xf32>

  %fwd:2 = enzyme.fwddiff @square(%x, %d) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<5xf32>, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)

  check.expect_almost_eq %fwd#0, %output : tensor<5xf32>
  check.expect_almost_eq %fwd#1, %expected : tensor<5xf32>

  %rev:2 = enzyme.autodiff @square(%x, %d) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<5xf32>, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)

  check.expect_almost_eq %rev#0, %output : tensor<5xf32>
  check.expect_almost_eq %rev#1, %expected : tensor<5xf32>

  func.return
}
