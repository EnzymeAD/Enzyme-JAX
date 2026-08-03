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

// FORWARD: module {
// FORWARD-NEXT:   func.func @main() {
// FORWARD-NEXT:     %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// FORWARD-NEXT:     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// FORWARD-NEXT:     %0:2 = enzyme.autodiff @f(%cst, %cst_0) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>]} : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
// FORWARD-NEXT:     check.expect_eq_const %0#0, dense<2.560000e+02> : tensor<f64>
// FORWARD-NEXT:     check.expect_eq_const %0#1, dense<1.024000e+03> : tensor<f64>
// FORWARD-NEXT:     return
// FORWARD-NEXT:   }
// FORWARD-NEXT:   func.func @f(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
// FORWARD-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// FORWARD-NEXT:     %c_0 = stablehlo.constant dense<1> : tensor<i64>
// FORWARD-NEXT:     %c_1 = stablehlo.constant dense<3> : tensor<i64>
// FORWARD-NEXT:     %0:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0, %iterArg_3 = %arg1) : tensor<i64>, tensor<f64>, tensor<f64>
// FORWARD-NEXT:     cond {
// FORWARD-NEXT:       %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// FORWARD-NEXT:       stablehlo.return %1 : tensor<i1>
// FORWARD-NEXT:     } do {
// FORWARD-NEXT:       %1 = stablehlo.multiply %iterArg_3, %iterArg_2 : tensor<f64>
// FORWARD-NEXT:       %2 = stablehlo.multiply %iterArg_3, %iterArg_2 : tensor<f64>
// FORWARD-NEXT:       %3 = arith.addf %1, %2 : tensor<f64>
// FORWARD-NEXT:       %4 = stablehlo.multiply %iterArg_2, %iterArg_2 : tensor<f64>
// FORWARD-NEXT:       %5 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// FORWARD-NEXT:       stablehlo.return %5, %4, %3 : tensor<i64>, tensor<f64>, tensor<f64>
// FORWARD-NEXT:     }
// FORWARD-NEXT:     return %0#1, %0#2 : tensor<f64>, tensor<f64>
// FORWARD-NEXT:   }
// FORWARD-NEXT: }

// REVERSE: module {
// REVERSE-NEXT:   func.func @main() {
// REVERSE-NEXT:     %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// REVERSE-NEXT:     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// REVERSE-NEXT:     %0:2 = call @diffef(%cst, %cst_0) : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
// REVERSE-NEXT:     check.expect_eq_const %0#0, dense<2.560000e+02> : tensor<f64>
// REVERSE-NEXT:     check.expect_eq_const %0#1, dense<1.024000e+03> : tensor<f64>
// REVERSE-NEXT:     return
// REVERSE-NEXT:   }
// REVERSE-NEXT:   func.func @f(%arg0: tensor<f64>) -> tensor<f64> {
// REVERSE-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// REVERSE-NEXT:     %c_0 = stablehlo.constant dense<1> : tensor<i64>
// REVERSE-NEXT:     %c_1 = stablehlo.constant dense<3> : tensor<i64>
// REVERSE-NEXT:     %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64>
// REVERSE-NEXT:     cond {
// REVERSE-NEXT:       %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// REVERSE-NEXT:       stablehlo.return %1 : tensor<i1>
// REVERSE-NEXT:     } do {
// REVERSE-NEXT:       %1 = stablehlo.multiply %iterArg_2, %iterArg_2 : tensor<f64>
// REVERSE-NEXT:       %2 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// REVERSE-NEXT:       stablehlo.return %2, %1 : tensor<i64>, tensor<f64>
// REVERSE-NEXT:     }
// REVERSE-NEXT:     return %0#1 : tensor<f64>
// REVERSE-NEXT:   }
// REVERSE-NEXT:   func.func private @diffef(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
// REVERSE-NEXT:     %c = stablehlo.constant dense<2> : tensor<i64>
// REVERSE-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<3xf64>
// REVERSE-NEXT:     %c_0 = stablehlo.constant dense<3> : tensor<i64>
// REVERSE-NEXT:     %c_1 = stablehlo.constant dense<1> : tensor<i64>
// REVERSE-NEXT:     %c_2 = stablehlo.constant dense<0> : tensor<i64>
// REVERSE-NEXT:     %cst_3 = arith.constant dense<0.000000e+00> : tensor<f64>
// REVERSE-NEXT:     %0:3 = stablehlo.while(%iterArg = %c_2, %iterArg_4 = %arg0, %iterArg_5 = %cst) : tensor<i64>, tensor<f64>, tensor<3xf64>
// REVERSE-NEXT:     cond {
// REVERSE-NEXT:       %4 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// REVERSE-NEXT:       stablehlo.return %4 : tensor<i1>
// REVERSE-NEXT:     } do {
// REVERSE-NEXT:       %4 = stablehlo.reshape %iterArg_4 : (tensor<f64>) -> tensor<1xf64>
// REVERSE-NEXT:       %5 = stablehlo.dynamic_update_slice %iterArg_5, %4, %iterArg : (tensor<3xf64>, tensor<1xf64>, tensor<i64>) -> tensor<3xf64>
// REVERSE-NEXT:       %6 = stablehlo.multiply %iterArg_4, %iterArg_4 : tensor<f64>
// REVERSE-NEXT:       %7 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// REVERSE-NEXT:       stablehlo.return %7, %6, %5 : tensor<i64>, tensor<f64>, tensor<3xf64>
// REVERSE-NEXT:     }
// REVERSE-NEXT:     %1 = arith.addf %arg1, %cst_3 : tensor<f64>
// REVERSE-NEXT:     %2:2 = stablehlo.while(%iterArg = %c_2, %iterArg_4 = %1) : tensor<i64>, tensor<f64>
// REVERSE-NEXT:     cond {
// REVERSE-NEXT:       %4 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// REVERSE-NEXT:       stablehlo.return %4 : tensor<i1>
// REVERSE-NEXT:     } do {
// REVERSE-NEXT:       %4 = stablehlo.subtract %c, %iterArg : tensor<i64>
// REVERSE-NEXT:       %5 = stablehlo.dynamic_slice %0#2, %4, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> tensor<1xf64>
// REVERSE-NEXT:       %6 = stablehlo.reshape %5 : (tensor<1xf64>) -> tensor<f64>
// REVERSE-NEXT:       %7 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// REVERSE-NEXT:       %8 = arith.addf %iterArg_4, %cst_3 : tensor<f64>
// REVERSE-NEXT:       %9 = stablehlo.multiply %8, %6 : tensor<f64>
// REVERSE-NEXT:       %10 = arith.addf %9, %cst_3 : tensor<f64>
// REVERSE-NEXT:       %11 = stablehlo.multiply %8, %6 : tensor<f64>
// REVERSE-NEXT:       %12 = arith.addf %10, %11 : tensor<f64>
// REVERSE-NEXT:       stablehlo.return %7, %12 : tensor<i64>, tensor<f64>
// REVERSE-NEXT:     }
// REVERSE-NEXT:     %3 = arith.addf %2#1, %cst_3 : tensor<f64>
// REVERSE-NEXT:     return %0#1, %3 : tensor<f64>, tensor<f64>
// REVERSE-NEXT:   }
// REVERSE-NEXT: }
