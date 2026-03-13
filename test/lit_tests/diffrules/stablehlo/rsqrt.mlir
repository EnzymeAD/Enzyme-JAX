// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=rsqrt outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=rsqrt outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @rsqrt(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = stablehlo.rsqrt %x : (tensor<2xf32>) -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:  func.func @rsqrt(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %cst = stablehlo.constant dense<-2.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %0 = stablehlo.sqrt %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %arg0, %0 : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.multiply %cst, %1 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.divide %arg1, %2 : tensor<2xf32>
// FORWARD-NEXT:    %4 = stablehlo.rsqrt %arg0 : tensor<2xf32>
// FORWARD-NEXT:    return %4, %3 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @rsqrt(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<-2.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = stablehlo.sqrt %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.multiply %arg0, %0 : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.multiply %cst, %1 : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.divide %arg1, %2 : tensor<2xf32>
// REVERSE-NEXT:    return %3 : tensor<2xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %x = stablehlo.constant dense<[4.0, 9.0]> : tensor<2xf32>
  %out = stablehlo.constant dense<[0.5, 0.33333334]> : tensor<2xf32>
  %expected = stablehlo.constant dense<[-0.0625, -0.018518518518511842]> : tensor<2xf32>

  %d = stablehlo.constant dense<1.0> : tensor<2xf32>

  %fwd:2 = enzyme.fwddiff @rsqrt(%x, %d) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd#0, %out : tensor<2xf32>
  check.expect_almost_eq %fwd#1, %expected : tensor<2xf32>

  %rev:2 = enzyme.autodiff @rsqrt(%x, %d) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %rev#0, %out : tensor<2xf32>
  check.expect_almost_eq %rev#1, %expected : tensor<2xf32>

  func.return
}
