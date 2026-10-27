// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=exp2 outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=exp2 outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// The interpreter run executes @main below: its check.expect_almost_eq ops
// compare the primal and the forward- and reverse-mode derivatives that enzyme
// produced against the expected values, and a mismatch makes
// stablehlo-translate exit non-zero.
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @exp2(%x : tensor<3xf32>) -> tensor<3xf32> {
  %y = stablehlo.exp2 %x : (tensor<3xf32>) -> tensor<3xf32>
  func.return %y : tensor<3xf32>
}

// FORWARD:  func.func @exp2(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
// FORWARD-NEXT:    %[[LN2:.+]] = stablehlo.constant dense<0.693147182> : tensor<3xf32>
// FORWARD-NEXT:    %[[E:.+]] = stablehlo.exp2 %arg0 : tensor<3xf32>
// FORWARD-NEXT:    %[[S:.+]] = stablehlo.multiply %[[E]], %[[LN2]] : tensor<3xf32>
// FORWARD-NEXT:    %[[D:.+]] = stablehlo.multiply %arg1, %[[S]] : tensor<3xf32>
// FORWARD-NEXT:    return %[[E]], %[[D]] : tensor<3xf32>, tensor<3xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @exp2(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
// REVERSE-NEXT:    %[[LN2:.+]] = stablehlo.constant dense<0.693147182> : tensor<3xf32>
// REVERSE-NEXT:    %[[E:.+]] = stablehlo.exp2 %arg0 : tensor<3xf32>
// REVERSE-NEXT:    %[[S:.+]] = stablehlo.multiply %[[E]], %[[LN2]] : tensor<3xf32>
// REVERSE-NEXT:    %[[D:.+]] = stablehlo.multiply %arg1, %[[S]] : tensor<3xf32>
// REVERSE-NEXT:    return %[[D]] : tensor<3xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %x = stablehlo.constant dense<[0.0, 1.0, -1.0]> : tensor<3xf32>
  %out = stablehlo.constant dense<[1.0, 2.0, 0.5]> : tensor<3xf32>
  // 2^x * ln(2)
  %expected = stablehlo.constant dense<[0.6931472, 1.3862944, 0.3465736]> : tensor<3xf32>

  %d = stablehlo.constant dense<1.0> : tensor<3xf32>

  %fwd:2 = enzyme.fwddiff @exp2(%x, %d) <{
    activity=[#enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)

  check.expect_almost_eq %fwd#0, %out : tensor<3xf32>
  check.expect_almost_eq %fwd#1, %expected : tensor<3xf32>

  %rev:2 = enzyme.autodiff @exp2(%x, %d) <{
    activity=[#enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)

  check.expect_almost_eq %rev#0, %out : tensor<3xf32>
  check.expect_almost_eq %rev#1, %expected : tensor<3xf32>

  func.return
}
