// RUN: enzymexlamlir-opt %s --remove-unnecessary-enzyme-ops | FileCheck %s

module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %cst_one  = stablehlo.constant dense<1.000000e+00> : tensor<3xf64>
    %cst_zero = stablehlo.constant dense<0.000000e+00> : tensor<3xf64>

    // Cache stores the running product computed in the forward loop.
    %cache = "enzyme.init"() : () -> !enzyme.Cache<tensor<3xf64>>

    // Forward loop (1 iteration): accumulate product, push each step to cache.
    %fwd:2 = stablehlo.while(%iterArg = %c0, %iterArg_prod = %cst_one) : tensor<i64>, tensor<3xf64>
     cond {
      %cond = stablehlo.compare LT, %iterArg, %c1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %prod = stablehlo.multiply %iterArg_prod, %arg0 : tensor<3xf64>
      "enzyme.push"(%cache, %prod) : (!enzyme.Cache<tensor<3xf64>>, tensor<3xf64>) -> ()
      %next = stablehlo.add %iterArg, %c1 : tensor<i64>
      stablehlo.return %next, %prod : tensor<i64>, tensor<3xf64>
    }

    // Backward loop (1 iteration): pop cached products and accumulate sum.
    %bwd:2 = stablehlo.while(%iterArg = %c0, %iterArg_sum = %cst_zero) : tensor<i64>, tensor<3xf64>
     cond {
      %cond = stablehlo.compare LT, %iterArg, %c1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %val = "enzyme.pop"(%cache) : (!enzyme.Cache<tensor<3xf64>>) -> tensor<3xf64>
      %sum = stablehlo.add %iterArg_sum, %val : tensor<3xf64>
      %next = stablehlo.add %iterArg, %c1 : tensor<i64>
      stablehlo.return %next, %sum : tensor<i64>, tensor<3xf64>
    }

    return %bwd#1 : tensor<3xf64>
  }
}

// CHECK:  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x3xf64>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<3xf64>
// CHECK-NEXT:    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<3xf64>
// CHECK-NEXT:    %0:3 = stablehlo.while(%iterArg = %c, %iterArg_3 = %cst_1, %iterArg_4 = %cst) : tensor<i64>, tensor<3xf64>, tensor<1x3xf64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %c_0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.reshape %iterArg_3 : (tensor<3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:      %3 = stablehlo.dynamic_update_slice %iterArg_4, %2, %iterArg, %c : (tensor<1x3xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
// CHECK-NEXT:      %4 = stablehlo.multiply %iterArg_3, %arg0 : tensor<3xf64>
// CHECK-NEXT:      %5 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %5, %4, %3 : tensor<i64>, tensor<3xf64>, tensor<1x3xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:3 = stablehlo.while(%iterArg = %c, %iterArg_3 = %cst_2, %iterArg_4 = %c) : tensor<i64>, tensor<3xf64>, tensor<i64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %c_0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.dynamic_slice %0#2, %iterArg_4, %c, sizes = [1, 3] : (tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
// CHECK-NEXT:      %3 = stablehlo.reshape %2 : (tensor<1x3xf64>) -> tensor<3xf64>
// CHECK-NEXT:      %4 = stablehlo.multiply %3, %arg0 : tensor<3xf64>
// CHECK-NEXT:      %5 = stablehlo.add %iterArg_3, %4 : tensor<3xf64>
// CHECK-NEXT:      %6 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      %7 = stablehlo.subtract %iterArg_4, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %6, %5, %7 : tensor<i64>, tensor<3xf64>, tensor<i64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %1#1 : tensor<3xf64>
// CHECK-NEXT:  }
