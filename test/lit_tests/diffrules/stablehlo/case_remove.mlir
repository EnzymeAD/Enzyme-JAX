// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --allow-unregistered-dialect | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%arg0: tensor<10xf32>, %index: tensor<i32>) -> tensor<10xf32> {
    %cst = stablehlo.constant dense<1.0> : tensor<10xf32>

    %0 = "stablehlo.case"(%index) ({
      %1 = stablehlo.add %arg0, %cst : tensor<10xf32>
      "stablehlo.return"(%1) : (tensor<10xf32>) -> ()
    }, {
      %2 = stablehlo.multiply %arg0, %cst : tensor<10xf32>
      %3 = stablehlo.multiply %2, %2 : tensor<10xf32>
      "stablehlo.return"(%3) : (tensor<10xf32>) -> ()
    }, {
      "stablehlo.return"(%cst) : (tensor<10xf32>) -> ()
    }) : (tensor<i32>) -> tensor<10xf32>

    return %0 : tensor<10xf32>
  }

  func.func @zmain2(%arg0: tensor<10xf32>, %index: tensor<i32>, %arg2: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<10xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<10xf32>
    %0 = "enzyme.init"() : () -> !enzyme.Gradient<tensor<10xf32>>
    "enzyme.set"(%0, %arg0) : (!enzyme.Gradient<tensor<10xf32>>, tensor<10xf32>) -> ()
    %1 = "stablehlo.case"(%index) ({
      "enzyme.set"(%0, %cst_0) : (!enzyme.Gradient<tensor<10xf32>>, tensor<10xf32>) -> ()
      %2 = "test.foo"(%arg0) : (tensor<10xf32>) -> tensor<10xf32>
      stablehlo.return %2 : tensor<10xf32>
    }, {
      "enzyme.set"(%0, %arg2) : (!enzyme.Gradient<tensor<10xf32>>, tensor<10xf32>) -> ()
      stablehlo.return %cst : tensor<10xf32>
    }, {
      %3 = "test.bar"(%arg2) : (tensor<10xf32>) -> tensor<10xf32>
      stablehlo.return %3 : tensor<10xf32>
    }) : (tensor<i32>) -> tensor<10xf32>
    %4 = "enzyme.get"(%0) : (!enzyme.Gradient<tensor<10xf32>>) -> tensor<10xf32>
    return %1, %4 : tensor<10xf32>, tensor<10xf32>
  }
}

// REVERSE:  func.func @zmain2(%arg0: tensor<10xf32>, %arg1: tensor<i32>, %arg2: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<10xf32>
// REVERSE-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
// REVERSE-NEXT:    %0:2 = "stablehlo.case"(%arg1) ({
// REVERSE-NEXT:      %1 = "test.foo"(%arg0) : (tensor<10xf32>) -> tensor<10xf32>
// REVERSE-NEXT:      stablehlo.return %1, %cst_0 : tensor<10xf32>, tensor<10xf32>
// REVERSE-NEXT:    }, {
// REVERSE-NEXT:      stablehlo.return %cst, %arg2 : tensor<10xf32>, tensor<10xf32>
// REVERSE-NEXT:    }, {
// REVERSE-NEXT:      %1 = "test.bar"(%arg2) : (tensor<10xf32>) -> tensor<10xf32>
// REVERSE-NEXT:      stablehlo.return %1, %arg0 : tensor<10xf32>, tensor<10xf32>
// REVERSE-NEXT:    }) : (tensor<i32>) -> (tensor<10xf32>, tensor<10xf32>)
// REVERSE-NEXT:    return %0#0, %0#1 : tensor<10xf32>, tensor<10xf32>
// REVERSE-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<i32>, %arg2: tensor<10xf32>) -> tensor<10xf32> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
// REVERSE-NEXT:    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<10xf32>
// REVERSE-NEXT:    %0:2 = "stablehlo.case"(%arg1) ({
// REVERSE-NEXT:      %2 = stablehlo.add %arg0, %cst_0 : tensor<10xf32>
// REVERSE-NEXT:      stablehlo.return %2, %cst : tensor<10xf32>, tensor<10xf32>
// REVERSE-NEXT:    }, {
// REVERSE-NEXT:      %2 = stablehlo.multiply %arg0, %arg0 : tensor<10xf32>
// REVERSE-NEXT:      stablehlo.return %2, %arg0 : tensor<10xf32>, tensor<10xf32>
// REVERSE-NEXT:    }, {
// REVERSE-NEXT:      stablehlo.return %cst_0, %cst : tensor<10xf32>, tensor<10xf32>
// REVERSE-NEXT:    }) : (tensor<i32>) -> (tensor<10xf32>, tensor<10xf32>)
// REVERSE-NEXT:    %1:4 = "stablehlo.case"(%arg1) ({
// REVERSE-NEXT:      stablehlo.return %cst, %arg2, %cst, %cst : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>
// REVERSE-NEXT:    }, {
// REVERSE-NEXT:      %2 = stablehlo.multiply %arg2, %0#1 : tensor<10xf32>
// REVERSE-NEXT:      %3 = stablehlo.add %2, %2 : tensor<10xf32>
// REVERSE-NEXT:      stablehlo.return %cst, %3, %cst, %cst : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>
// REVERSE-NEXT:    }, {
// REVERSE-NEXT:      stablehlo.return %cst, %cst, %cst, %cst : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>
// REVERSE-NEXT:    }) : (tensor<i32>) -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>)
// REVERSE-NEXT:    return %1#1 : tensor<10xf32>
// REVERSE-NEXT:  }
