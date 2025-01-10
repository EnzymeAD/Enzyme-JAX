// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --allow-unregistered-dialect | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%arg0: tensor<10xf32>, %pred: tensor<i1>) -> tensor<10xf32> {
    %cst = stablehlo.constant dense<1.0> : tensor<10xf32>

    %0 = "stablehlo.if"(%pred) ({
      %1 = stablehlo.multiply %arg0, %cst : tensor<10xf32>
      %2 = stablehlo.multiply %1, %1 : tensor<10xf32>
      "stablehlo.return"(%2) : (tensor<10xf32>) -> ()
    }, {
      "stablehlo.return"(%cst) : (tensor<10xf32>) -> ()
    }) : (tensor<i1>) -> tensor<10xf32>

    return %0 : tensor<10xf32>
  }
  func.func @zmain2(%arg0: tensor<10xf32>, %arg1: tensor<i1>, %arg2: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<10xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<10xf32>
    %0 = "enzyme.init"() : () -> !enzyme.Gradient<tensor<10xf32>>
    %8 = "stablehlo.if"(%arg1) ({
      "enzyme.set"(%0, %cst_0) : (!enzyme.Gradient<tensor<10xf32>>, tensor<10xf32>) -> ()
      %13 = "test.foo"(%arg0) : (tensor<10xf32>) -> tensor<10xf32>
      stablehlo.return %13 : tensor<10xf32>
    }, {
      "enzyme.set"(%0, %arg2) : (!enzyme.Gradient<tensor<10xf32>>, tensor<10xf32>) -> ()
      stablehlo.return %cst : tensor<10xf32>
    }) : (tensor<i1>) -> tensor<10xf32>
    %9 = "enzyme.get"(%0) : (!enzyme.Gradient<tensor<10xf32>>) -> tensor<10xf32>
    return %8, %9 : tensor<10xf32>, tensor<10xf32>
  }
}

// REVERSE:  func.func @zmain2(%arg0: tensor<10xf32>, %arg1: tensor<i1>, %arg2: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<10xf32>
// REVERSE-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
// REVERSE-NEXT:    %0 = stablehlo.select %arg1, %cst_0, %arg2 : tensor<i1>, tensor<10xf32>
// REVERSE-NEXT:    %1 = "stablehlo.if"(%arg1) ({
// REVERSE-NEXT:      %2 = "test.foo"(%arg0) : (tensor<10xf32>) -> tensor<10xf32>
// REVERSE-NEXT:      stablehlo.return %2 : tensor<10xf32>
// REVERSE-NEXT:    }, {
// REVERSE-NEXT:      stablehlo.return %cst : tensor<10xf32>
// REVERSE-NEXT:    }) : (tensor<i1>) -> tensor<10xf32>
// REVERSE-NEXT:    return %1, %0 : tensor<10xf32>, tensor<10xf32>
// REVERSE-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<i1>, %arg2: tensor<10xf32>) -> tensor<10xf32> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
// REVERSE-NEXT:    %0 = "stablehlo.if"(%arg1) ({
// REVERSE-NEXT:      %1 = stablehlo.multiply %arg2, %arg0 : tensor<10xf32>
// REVERSE-NEXT:      %2 = stablehlo.add %1, %1 : tensor<10xf32>
// REVERSE-NEXT:      stablehlo.return %2 : tensor<10xf32>
// REVERSE-NEXT:    }, {
// REVERSE-NEXT:      stablehlo.return %cst : tensor<10xf32>
// REVERSE-NEXT:    }) : (tensor<i1>) -> tensor<10xf32>
// REVERSE-NEXT:    return %0 : tensor<10xf32>
// REVERSE-NEXT:  }
