// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=negate outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=negate outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @negate(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = stablehlo.negate %x : (tensor<2xf32>) -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:  func.func @negate(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.negate %arg1 : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.negate %arg0 : tensor<2xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @negate(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %0 = stablehlo.negate %arg1 : tensor<2xf32>
// REVERSE-NEXT:    return %0 : tensor<2xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %x = stablehlo.constant dense<[1.0, -2.0]> : tensor<2xf32>
  %out = stablehlo.constant dense<[-1.0, 2.0]> : tensor<2xf32>
  %expected = stablehlo.constant dense<[-1.0, -1.0]> : tensor<2xf32>

  %dx = stablehlo.constant dense<1.0> : tensor<2xf32>

  %fwd:2 = enzyme.fwddiff @negate(%x, %dx) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd#0, %out : tensor<2xf32>
  check.expect_almost_eq %fwd#1, %expected : tensor<2xf32>

  %rev:2 = enzyme.autodiff @negate(%x, %dx) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %rev#0, %out : tensor<2xf32>
  check.expect_almost_eq %rev#1, %expected : tensor<2xf32>

  func.return
}
