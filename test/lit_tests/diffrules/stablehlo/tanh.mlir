// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=tanh outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=tanh outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @tanh(%x : tensor<3xf32>) -> tensor<3xf32> {
  %y = stablehlo.tanh %x : (tensor<3xf32>) -> tensor<3xf32>
  func.return %y : tensor<3xf32>
}

// FORWARD:  func.func @tanh(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
// FORWARD-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// FORWARD-NEXT:    %0 = stablehlo.tanh %arg0 : tensor<3xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %0, %0 : tensor<3xf32>
// FORWARD-NEXT:    %2 = stablehlo.subtract %cst, %1 : tensor<3xf32>
// FORWARD-NEXT:    %3 = stablehlo.multiply %arg1, %2 : tensor<3xf32>
// FORWARD-NEXT:    return %0, %3 : tensor<3xf32>, tensor<3xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @tanh(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// REVERSE-NEXT:    %0 = stablehlo.tanh %arg0 : tensor<3xf32>
// REVERSE-NEXT:    %1 = stablehlo.multiply %0, %0 : tensor<3xf32>
// REVERSE-NEXT:    %2 = stablehlo.subtract %cst, %1 : tensor<3xf32>
// REVERSE-NEXT:    %3 = stablehlo.multiply %arg1, %2 : tensor<3xf32>
// REVERSE-NEXT:    return %3 : tensor<3xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %x = stablehlo.constant dense<[0.0, 1.0, -1.0]> : tensor<3xf32>
  %out = stablehlo.constant dense<[0.0, 0.7615941559557649, -0.7615941559557649]> : tensor<3xf32>
  %expected = stablehlo.constant dense<[1.0, 0.41997434161404607, 0.4199743416139812]> : tensor<3xf32>

  %dx = stablehlo.constant dense<1.0> : tensor<3xf32>

  %fwd:2 = enzyme.fwddiff @tanh(%x, %dx) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)

  check.expect_almost_eq %fwd#0, %out : tensor<3xf32>
  check.expect_almost_eq %fwd#1, %expected : tensor<3xf32>

  %rev:2 = enzyme.autodiff @tanh(%x, %dx) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)

  check.expect_almost_eq %rev#0, %out : tensor<3xf32>
  check.expect_almost_eq %rev#1, %expected : tensor<3xf32>

  func.return
}
