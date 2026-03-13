// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=sqrt outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=sqrt outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @sqrt(%x : tensor<3xf32>) -> tensor<3xf32> {
  %y = stablehlo.sqrt %x : (tensor<3xf32>) -> tensor<3xf32>
  func.return %y : tensor<3xf32>
}

// FORWARD:  func.func @sqrt(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
// FORWARD-NEXT:    %cst = stablehlo.constant dense<2.000000e+00> : tensor<3xf32>
// FORWARD-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
// FORWARD-NEXT:    %0 = stablehlo.compare  EQ, %arg0, %cst_0 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
// FORWARD-NEXT:    %1 = stablehlo.sqrt %arg0 : tensor<3xf32>
// FORWARD-NEXT:    %2 = stablehlo.multiply %cst, %1 : tensor<3xf32>
// FORWARD-NEXT:    %3 = stablehlo.divide %arg1, %2 : tensor<3xf32>
// FORWARD-NEXT:    %4 = stablehlo.select %0, %cst_0, %3 : tensor<3xi1>, tensor<3xf32>
// FORWARD-NEXT:    return %1, %4 : tensor<3xf32>, tensor<3xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @sqrt(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<2.000000e+00> : tensor<3xf32>
// REVERSE-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
// REVERSE-NEXT:    %0 = stablehlo.compare  EQ, %arg0, %cst_0 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
// REVERSE-NEXT:    %1 = stablehlo.sqrt %arg0 : tensor<3xf32>
// REVERSE-NEXT:    %2 = stablehlo.multiply %cst, %1 : tensor<3xf32>
// REVERSE-NEXT:    %3 = stablehlo.divide %arg1, %2 : tensor<3xf32>
// REVERSE-NEXT:    %4 = stablehlo.select %0, %cst_0, %3 : tensor<3xi1>, tensor<3xf32>
// REVERSE-NEXT:    return %4 : tensor<3xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %x = stablehlo.constant dense<[4.0, 9.0, 0.0]> : tensor<3xf32>
  %out = stablehlo.constant dense<[2.0, 3.0, 0.0]> : tensor<3xf32>
  %expected = stablehlo.constant dense<[0.25, 0.16666667, 0.0]> : tensor<3xf32>

  %d = stablehlo.constant dense<1.0> : tensor<3xf32>

  %fwd:2 = enzyme.fwddiff @sqrt(%x, %d) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)

  check.expect_almost_eq %fwd#0, %out : tensor<3xf32>
  check.expect_almost_eq %fwd#1, %expected : tensor<3xf32>

  %rev:2 = enzyme.autodiff @sqrt(%x, %d) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)

  check.expect_almost_eq %rev#0, %out : tensor<3xf32>
  check.expect_almost_eq %rev#1, %expected : tensor<3xf32>

  func.return
}
