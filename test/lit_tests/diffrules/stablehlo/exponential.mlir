// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=exponential outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=exponential outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @exponential(%x : tensor<3xf32>) -> tensor<3xf32> {
  %y = stablehlo.exponential %x : (tensor<3xf32>) -> tensor<3xf32>
  func.return %y : tensor<3xf32>
}

// FORWARD:  func.func @exponential(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.exponential %arg0 : tensor<3xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %arg1, %0 : tensor<3xf32>
// FORWARD-NEXT:    return %0, %1 : tensor<3xf32>, tensor<3xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @exponential(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
// REVERSE-NEXT:    %0 = stablehlo.exponential %arg0 : tensor<3xf32>
// REVERSE-NEXT:    %1 = stablehlo.multiply %arg1, %0 : tensor<3xf32>
// REVERSE-NEXT:    return %1 : tensor<3xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %x = stablehlo.constant dense<[0.0, 1.0, -1.0]> : tensor<3xf32>
  %out = stablehlo.constant dense<[1.0, 2.7182817, 0.36787945]> : tensor<3xf32>
  %expected = stablehlo.constant dense<[1.0, 2.7182817, 0.36787945]> : tensor<3xf32>

  %d = stablehlo.constant dense<1.0> : tensor<3xf32>

  %fwd:2 = enzyme.fwddiff @exponential(%x, %d) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)

  check.expect_almost_eq %fwd#0, %out : tensor<3xf32>
  check.expect_almost_eq %fwd#1, %expected : tensor<3xf32>

  %rev:2 = enzyme.autodiff @exponential(%x, %d) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)

  check.expect_almost_eq %rev#0, %out : tensor<3xf32>
  check.expect_almost_eq %rev#1, %expected : tensor<3xf32>

  func.return
}
