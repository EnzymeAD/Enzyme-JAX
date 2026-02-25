// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=logistic outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=logistic outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

func.func @logistic(%x : tensor<4xf32>) -> tensor<4xf32> {
  %y = stablehlo.logistic %x : (tensor<4xf32>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// FORWARD:  func.func @logistic(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// FORWARD-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
// FORWARD-NEXT:    %0 = stablehlo.logistic %arg0 : tensor<4xf32>
// FORWARD-NEXT:    %1 = stablehlo.subtract %cst, %0 : tensor<4xf32>
// FORWARD-NEXT:    %2 = stablehlo.multiply %0, %1 : tensor<4xf32>
// FORWARD-NEXT:    %3 = stablehlo.multiply %arg1, %2 : tensor<4xf32>
// FORWARD-NEXT:    return %0, %3 : tensor<4xf32>, tensor<4xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @logistic(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
// REVERSE-NEXT:    %0 = stablehlo.logistic %arg0 : tensor<4xf32>
// REVERSE-NEXT:    %1 = stablehlo.subtract %cst, %0 : tensor<4xf32>
// REVERSE-NEXT:    %2 = stablehlo.multiply %0, %1 : tensor<4xf32>
// REVERSE-NEXT:    %3 = stablehlo.multiply %arg1, %2 : tensor<4xf32>
// REVERSE-NEXT:    return %3 : tensor<4xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %input = stablehlo.constant dense<[-1.0, 0.0, 1.0, 10.0]> : tensor<4xf32>
  %output = stablehlo.constant dense<[0.2689414213699951, 0.5, 0.7310585786300049, 0.9999546021312976]> : tensor<4xf32>
  %expected = stablehlo.constant dense<[0.19661193324148166, 0.25, 0.19661193324150486, 4.5395807738167505e-5]> : tensor<4xf32>

  %dinput = stablehlo.constant dense<1.0> : tensor<4xf32>

  // fwd diff
  %fwd_res:2 = enzyme.fwddiff @logistic(%input, %dinput) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)

  check.expect_almost_eq %fwd_res#0, %output : tensor<4xf32>
  check.expect_almost_eq %fwd_res#1, %expected : tensor<4xf32>

  // rev diff
  %rev_res:2 = enzyme.autodiff @logistic(%input, %dinput) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)

  check.expect_almost_eq %rev_res#0, %output : tensor<4xf32>
  check.expect_almost_eq %rev_res#1, %expected : tensor<4xf32>

  func.return
}
