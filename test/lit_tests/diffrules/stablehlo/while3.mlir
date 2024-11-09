// RUN: enzymexlamlir-opt %s --enzyme --canonicalize | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --enzyme-hlo-unroll --enzyme-hlo-opt --enzyme-hlo-unroll --remove-unnecessary-enzyme-ops --enzyme-hlo-opt | stablehlo-translate - --interpret

module {

  func.func @main() {
    %c2 = stablehlo.constant dense<2.0> : tensor<f64>
    %cst = stablehlo.constant dense<1.0> : tensor<f64>
    %0:2 = enzyme.autodiff @f(%c2, %cst) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    check.expect_eq_const %0#0, dense<256.0> : tensor<f64>
    check.expect_eq_const %0#1, dense<1024.0> : tensor<f64>
    return
  }

  func.func @f(%x: tensor<f64>) -> tensor<f64> {
    %init_i = stablehlo.constant dense<0> : tensor<i64>
    %init_sum = stablehlo.constant dense<0.0> : tensor<f64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %ten = stablehlo.constant dense<3> : tensor<i64>
    %results0, %results1 = "stablehlo.while"(%init_i, %x) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<f64>):
      %cond = "stablehlo.compare"(%arg0, %ten) {
        comparison_direction = #stablehlo<comparison_direction LT>
      } : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<f64>):
      %new_sum = stablehlo.multiply %arg1, %arg1 : tensor<f64>
      %new_i = stablehlo.add %arg0, %one : tensor<i64>
      stablehlo.return %new_i, %new_sum : tensor<i64>, tensor<f64>
    }) : (tensor<i64>, tensor<f64>) -> (tensor<i64>, tensor<f64>)
    return %results1 : tensor<f64>
  }
}

// CHECK:  func.func private @diffef(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = "enzyme.init"() : () -> !enzyme.Gradient<tensor<f64>>
// CHECK-NEXT:    "enzyme.set"(%0, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
// CHECK-NEXT:    %1 = "enzyme.init"() : () -> !enzyme.Gradient<tensor<f64>>
// CHECK-NEXT:    "enzyme.set"(%1, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
// CHECK-NEXT:    %2 = "enzyme.init"() : () -> !enzyme.Cache<tensor<f64>>
// CHECK-NEXT:    %3 = "enzyme.init"() : () -> !enzyme.Cache<tensor<f64>>
// CHECK-NEXT:    %4 = "enzyme.init"() : () -> !enzyme.Gradient<tensor<f64>>
// CHECK-NEXT:    "enzyme.set"(%4, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
// CHECK-NEXT:    %5 = "enzyme.init"() : () -> !enzyme.Gradient<tensor<f64>>
// CHECK-NEXT:    "enzyme.set"(%5, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
// CHECK-NEXT:    %6:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %14 = stablehlo.compare  LT, %iterArg, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %14 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      "enzyme.push"(%3, %iterArg_2) : (!enzyme.Cache<tensor<f64>>, tensor<f64>) -> ()
// CHECK-NEXT:      "enzyme.push"(%2, %iterArg_2) : (!enzyme.Cache<tensor<f64>>, tensor<f64>) -> ()
// CHECK-NEXT:      %14 = stablehlo.multiply %iterArg_2, %iterArg_2 : tensor<f64>
// CHECK-NEXT:      %15 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %15, %14 : tensor<i64>, tensor<f64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %7 = "enzyme.get"(%5) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
// CHECK-NEXT:    %8 = arith.addf %7, %arg1 : tensor<f64>
// CHECK-NEXT:    "enzyme.set"(%5, %8) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
// CHECK-NEXT:    %9 = "enzyme.get"(%5) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
// CHECK-NEXT:    "enzyme.set"(%5, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
// CHECK-NEXT:    %10:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %9) : tensor<i64>, tensor<f64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %14 = stablehlo.compare  LT, %iterArg, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %14 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %14 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      "enzyme.set"(%4, %iterArg_2) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
// CHECK-NEXT:      %15 = "enzyme.get"(%4) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
// CHECK-NEXT:      "enzyme.set"(%4, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
// CHECK-NEXT:      %16 = "enzyme.pop"(%3) : (!enzyme.Cache<tensor<f64>>) -> tensor<f64>
// CHECK-NEXT:      %17 = "enzyme.pop"(%2) : (!enzyme.Cache<tensor<f64>>) -> tensor<f64>
// CHECK-NEXT:      %18 = stablehlo.multiply %15, %17 : tensor<f64>
// CHECK-NEXT:      %19 = "enzyme.get"(%1) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
// CHECK-NEXT:      %20 = arith.addf %19, %18 : tensor<f64>
// CHECK-NEXT:      "enzyme.set"(%1, %20) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
// CHECK-NEXT:      %21 = stablehlo.multiply %15, %16 : tensor<f64>
// CHECK-NEXT:      %22 = "enzyme.get"(%1) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
// CHECK-NEXT:      %23 = arith.addf %22, %21 : tensor<f64>
// CHECK-NEXT:      "enzyme.set"(%1, %23) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
// CHECK-NEXT:      %24 = "enzyme.get"(%1) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
// CHECK-NEXT:      "enzyme.set"(%1, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
// CHECK-NEXT:      stablehlo.return %14, %24 : tensor<i64>, tensor<f64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %11 = "enzyme.get"(%0) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
// CHECK-NEXT:    %12 = arith.addf %11, %10#1 : tensor<f64>
// CHECK-NEXT:    "enzyme.set"(%0, %12) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
// CHECK-NEXT:    %13 = "enzyme.get"(%0) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
// CHECK-NEXT:    return %6#1, %13 : tensor<f64>, tensor<f64>
// CHECK-NEXT:  }
