// RUN: enzymexlamlir-opt %s --remove-unnecessary-enzyme-ops --enzyme-hlo-unroll --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

module {
  func.func @sum() -> tensor<i64> {
    %cst_0 = stablehlo.constant dense<0> : tensor<i64>
    %cst_1 = stablehlo.constant dense<1> : tensor<i64>
    %cst_10 = stablehlo.constant dense<10> : tensor<i64>
    %cache = "enzyme.init"() : () -> (!enzyme.Cache<tensor<i64>>)

    %0 = stablehlo.while(%iterArg = %cst_0) : tensor<i64>
     cond {
      %2 = stablehlo.compare LT, %iterArg, %cst_10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %newIter = stablehlo.add %iterArg, %cst_1 : tensor<i64>
      "enzyme.push"(%cache, %iterArg) : (!enzyme.Cache<tensor<i64>>, tensor<i64>) -> ()
      stablehlo.return %newIter : tensor<i64>
    }

    %1:2 = stablehlo.while(%iterArg = %cst_0, %iterArg2 = %cst_0) : tensor<i64>, tensor<i64>
     cond {
      %2 = stablehlo.compare LT, %iterArg, %cst_10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %newIter = stablehlo.add %iterArg, %cst_1 : tensor<i64>
      %pop = "enzyme.pop"(%cache) : (!enzyme.Cache<tensor<i64>>) -> tensor<i64>
      %newSum = stablehlo.add %iterArg2, %pop : tensor<i64>
      stablehlo.return %newIter, %newSum : tensor<i64>, tensor<i64>
    }
    return %1#1 : tensor<i64>
  }

  func.func @main() {
    %0 = func.call @sum() : () -> tensor<i64>
    check.expect_eq_const %0, dense<45> : tensor<i64>
    return
  }
}

// CHECK:  func.func @sum() -> tensor<i64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<45> : tensor<i64>
// CHECK-NEXT:    return %c : tensor<i64>
// CHECK-NEXT:  }
