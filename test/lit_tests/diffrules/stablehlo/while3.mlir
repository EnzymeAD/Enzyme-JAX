// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=f outfn= argTys=enzyme_dup retTys=enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE
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

// FORWARD:  func.func @f(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
// FORWARD-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// FORWARD-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// FORWARD-NEXT:    %c_1 = stablehlo.constant dense<3> : tensor<i64>
// FORWARD-NEXT:    %0:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0, %iterArg_3 = %arg1) : tensor<i64>, tensor<f64>, tensor<f64>
// FORWARD-NEXT:     cond {
// FORWARD-NEXT:      %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// FORWARD-NEXT:      stablehlo.return %1 : tensor<i1>
// FORWARD-NEXT:    } do {
// FORWARD-NEXT:      %1 = stablehlo.multiply %iterArg_3, %iterArg_2 : tensor<f64>
// FORWARD-NEXT:      %2 = stablehlo.multiply %iterArg_3, %iterArg_2 : tensor<f64>
// FORWARD-NEXT:      %3 = arith.addf %1, %2 : tensor<f64>
// FORWARD-NEXT:      %4 = stablehlo.multiply %iterArg_2, %iterArg_2 : tensor<f64>
// FORWARD-NEXT:      %5 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// FORWARD-NEXT:      stablehlo.return %5, %4, %3 : tensor<i64>, tensor<f64>, tensor<f64>
// FORWARD-NEXT:    }
// FORWARD-NEXT:    return %0#1, %0#2 : tensor<f64>, tensor<f64>
// FORWARD-NEXT:  }

// REVERSE:  func.func private @diffef(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
// REVERSE-NEXT:    %c = stablehlo.constant dense<3> : tensor<i64>
// REVERSE-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// REVERSE-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// REVERSE-NEXT:    %0 = "enzyme.init"() : () -> !enzyme.Cache<tensor<f64>>
// REVERSE-NEXT:    %1 = "enzyme.init"() : () -> !enzyme.Cache<tensor<f64>>
// REVERSE-NEXT:    %2 = stablehlo.subtract %c, %c_1 : tensor<i64>
// REVERSE-NEXT:    %3 = stablehlo.divide %2, %c_0 : tensor<i64>
// REVERSE-NEXT:    %4:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64>
// REVERSE-NEXT:     cond {
// REVERSE-NEXT:      %8 = stablehlo.compare  LT, %iterArg, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
// REVERSE-NEXT:      stablehlo.return %8 : tensor<i1>
// REVERSE-NEXT:    } do {
// REVERSE-NEXT:      "enzyme.push"(%1, %iterArg_2) : (!enzyme.Cache<tensor<f64>>, tensor<f64>) -> ()
// REVERSE-NEXT:      "enzyme.push"(%0, %iterArg_2) : (!enzyme.Cache<tensor<f64>>, tensor<f64>) -> ()
// REVERSE-NEXT:      %8 = stablehlo.multiply %iterArg_2, %iterArg_2 : tensor<f64>
// REVERSE-NEXT:      %9 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// REVERSE-NEXT:      stablehlo.return %9, %8 : tensor<i64>, tensor<f64>
// REVERSE-NEXT:    }
// REVERSE-NEXT:    %5 = arith.addf %arg1, %cst : tensor<f64>
// REVERSE-NEXT:    %6:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %5) : tensor<i64>, tensor<f64>
// REVERSE-NEXT:     cond {
// REVERSE-NEXT:      %8 = stablehlo.compare  LT, %iterArg, %3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// REVERSE-NEXT:      stablehlo.return %8 : tensor<i1>
// REVERSE-NEXT:    } do {
// REVERSE-NEXT:      %8 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// REVERSE-NEXT:      %9 = "enzyme.pop"(%1) : (!enzyme.Cache<tensor<f64>>) -> tensor<f64>
// REVERSE-NEXT:      %10 = "enzyme.pop"(%0) : (!enzyme.Cache<tensor<f64>>) -> tensor<f64>
// REVERSE-NEXT:      %11 = stablehlo.multiply %iterArg_2, %10 : tensor<f64>
// REVERSE-NEXT:      %12 = arith.addf %11, %cst : tensor<f64>
// REVERSE-NEXT:      %13 = stablehlo.multiply %iterArg_2, %9 : tensor<f64>
// REVERSE-NEXT:      %14 = arith.addf %12, %13 : tensor<f64>
// REVERSE-NEXT:      stablehlo.return %8, %14 : tensor<i64>, tensor<f64>
// REVERSE-NEXT:    }
// REVERSE-NEXT:    %7 = arith.addf %6#1, %cst : tensor<f64>
// REVERSE-NEXT:    return %4#1, %7 : tensor<f64>, tensor<f64>
// REVERSE-NEXT:  }
