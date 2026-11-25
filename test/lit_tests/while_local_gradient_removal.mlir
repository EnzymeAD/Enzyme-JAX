// RUN: enzymexlamlir-opt %s --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s

// A gradient created inside the loop body lives for one iteration only. It
// must not become a loop-carried value read before the loop: its enzyme.init
// is in the body and would not dominate the hoisted enzyme.get. The update
// inside the stablehlo.if keeps it from being simplified away before the
// while remover sees it, as in the reverse body of a checkpointed loop.

module {
  func.func @main(%arg0: tensor<4xf32>, %pred: tensor<i1>) -> tensor<4xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c10 = stablehlo.constant dense<10> : tensor<i64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>

    %0:2 = stablehlo.while(%iv = %c0, %acc = %arg0) : tensor<i64>, tensor<4xf32>
     cond {
      %1 = stablehlo.compare LT, %iv, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %g = "enzyme.init"() : () -> !enzyme.Gradient<tensor<4xf32>>
      "enzyme.set"(%g, %zero) : (!enzyme.Gradient<tensor<4xf32>>, tensor<4xf32>) -> ()

      "stablehlo.if"(%pred) ({
        %v0 = "enzyme.get"(%g) : (!enzyme.Gradient<tensor<4xf32>>) -> tensor<4xf32>
        %v1 = stablehlo.add %v0, %acc : tensor<4xf32>
        "enzyme.set"(%g, %v1) : (!enzyme.Gradient<tensor<4xf32>>, tensor<4xf32>) -> ()
        stablehlo.return
      }, {
        stablehlo.return
      }) : (tensor<i1>) -> ()

      %2 = "enzyme.get"(%g) : (!enzyme.Gradient<tensor<4xf32>>) -> tensor<4xf32>
      "enzyme.set"(%g, %zero) : (!enzyme.Gradient<tensor<4xf32>>, tensor<4xf32>) -> ()
      %3 = stablehlo.add %iv, %c1 : tensor<i64>
      stablehlo.return %3, %2 : tensor<i64>, tensor<4xf32>
    }

    return %0#1 : tensor<4xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<i1>) -> tensor<4xf32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<4xf32>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %1 = "stablehlo.if"(%arg1) ({
// CHECK-NEXT:        %3 = stablehlo.add %cst, %iterArg_2 : tensor<4xf32>
// CHECK-NEXT:        stablehlo.return %3 : tensor<4xf32>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %cst : tensor<4xf32>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<4xf32>
// CHECK-NEXT:      %2 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %2, %1 : tensor<i64>, tensor<4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0#1 : tensor<4xf32>
// CHECK-NEXT:  }
