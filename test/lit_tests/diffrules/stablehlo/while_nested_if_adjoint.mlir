// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --canonicalize --cse | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-ml --inline --enzyme-hlo-opt --drop-unsupported-attributes --symbol-dce | stablehlo-translate --interpret

// Reverse mode of a counted stablehlo.while whose body holds a stablehlo.if,
// one of whose branches holds a second stablehlo.if (or stablehlo.case):
//
//   x = 1; acc = 0
//   for i in 0..6:
//     if kind[i] == 1: acc += x
//     else:            x = n[i] > 1 ? x * a : x + 1
//
// With kind = [0, 0, 1, 0, 0, 1] and n = [2, 1, 0, 2, 1, 0] this computes
// acc = a^2 + 2a + 2, so d acc / d a = 2a + 2 = 7 at a = 2.5.
//
// The inner op's results are defined one region below the loop body. The
// loop re-zeroes the adjoints of its body's own values every iteration, but
// not of these, and the reverse if/case read its result adjoint in the
// branches without ever zeroing it: a later iteration consumed an earlier
// iteration's adjoint again and the derivative came out as 23 instead of 7.
// (Six iterations: shorter constant-trip loops are unrolled before
// differentiation.)

module {
  func.func private @nested_if(%a: tensor<f64>) -> tensor<f64> {
    %kind = stablehlo.constant dense<[0, 0, 1, 0, 0, 1]> : tensor<6xi64>
    %n = stablehlo.constant dense<[2, 1, 0, 2, 1, 0]> : tensor<6xi64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c6 = stablehlo.constant dense<6> : tensor<i64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %one = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %r:3 = stablehlo.while(%i = %c0, %x = %one, %acc = %zero) : tensor<i64>, tensor<f64>, tensor<f64> attributes {enzyme.disable_mincut}
    cond {
      %lt = stablehlo.compare LT, %i, %c6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %lt : tensor<i1>
    } do {
      %k1 = stablehlo.dynamic_slice %kind, %i, sizes = [1] : (tensor<6xi64>, tensor<i64>) -> tensor<1xi64>
      %k = stablehlo.reshape %k1 : (tensor<1xi64>) -> tensor<i64>
      %m1 = stablehlo.dynamic_slice %n, %i, sizes = [1] : (tensor<6xi64>, tensor<i64>) -> tensor<1xi64>
      %m = stablehlo.reshape %m1 : (tensor<1xi64>) -> tensor<i64>
      %is_acc = stablehlo.compare EQ, %k, %c1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %out:2 = "stablehlo.if"(%is_acc) ({
        %acc1 = stablehlo.add %acc, %x : tensor<f64>
        stablehlo.return %x, %acc1 : tensor<f64>, tensor<f64>
      }, {
        %many = stablehlo.compare GT, %m, %c1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %x2 = "stablehlo.if"(%many) ({
          %y = stablehlo.multiply %x, %a : tensor<f64>
          stablehlo.return %y : tensor<f64>
        }, {
          %y = stablehlo.add %x, %one : tensor<f64>
          stablehlo.return %y : tensor<f64>
        }) : (tensor<i1>) -> tensor<f64>
        stablehlo.return %x2, %acc : tensor<f64>, tensor<f64>
      }) : (tensor<i1>) -> (tensor<f64>, tensor<f64>)
      %inext = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %inext, %out#0, %out#1 : tensor<i64>, tensor<f64>, tensor<f64>
    }
    return %r#2 : tensor<f64>
  }

  // The same recurrence with the inner branch as a stablehlo.case.
  func.func private @nested_case(%a: tensor<f64>) -> tensor<f64> {
    %kind = stablehlo.constant dense<[0, 0, 1, 0, 0, 1]> : tensor<6xi64>
    %n = stablehlo.constant dense<[2, 1, 0, 2, 1, 0]> : tensor<6xi64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c6 = stablehlo.constant dense<6> : tensor<i64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %one = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %r:3 = stablehlo.while(%i = %c0, %x = %one, %acc = %zero) : tensor<i64>, tensor<f64>, tensor<f64> attributes {enzyme.disable_mincut}
    cond {
      %lt = stablehlo.compare LT, %i, %c6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %lt : tensor<i1>
    } do {
      %k1 = stablehlo.dynamic_slice %kind, %i, sizes = [1] : (tensor<6xi64>, tensor<i64>) -> tensor<1xi64>
      %k = stablehlo.reshape %k1 : (tensor<1xi64>) -> tensor<i64>
      %m1 = stablehlo.dynamic_slice %n, %i, sizes = [1] : (tensor<6xi64>, tensor<i64>) -> tensor<1xi64>
      %m = stablehlo.reshape %m1 : (tensor<1xi64>) -> tensor<i64>
      %is_acc = stablehlo.compare EQ, %k, %c1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %out:2 = "stablehlo.if"(%is_acc) ({
        %acc1 = stablehlo.add %acc, %x : tensor<f64>
        stablehlo.return %x, %acc1 : tensor<f64>, tensor<f64>
      }, {
        %many = stablehlo.compare GT, %m, %c1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %index = stablehlo.convert %many : (tensor<i1>) -> tensor<i32>
        %x2 = "stablehlo.case"(%index) ({
          %y = stablehlo.add %x, %one : tensor<f64>
          stablehlo.return %y : tensor<f64>
        }, {
          %y = stablehlo.multiply %x, %a : tensor<f64>
          stablehlo.return %y : tensor<f64>
        }) : (tensor<i32>) -> tensor<f64>
        stablehlo.return %x2, %acc : tensor<f64>, tensor<f64>
      }) : (tensor<i1>) -> (tensor<f64>, tensor<f64>)
      %inext = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %inext, %out#0, %out#1 : tensor<i64>, tensor<f64>, tensor<f64>
    }
    return %r#2 : tensor<f64>
  }

  func.func @main() {
    %a = stablehlo.constant dense<2.500000e+00> : tensor<f64>
    %seed = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %expected = stablehlo.constant dense<7.000000e+00> : tensor<f64>
    %0 = enzyme.autodiff @nested_if(%a, %seed) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %1 = enzyme.autodiff @nested_case(%a, %seed) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<f64>, tensor<f64>) -> tensor<f64>
    check.expect_close %0, %expected, max_ulp_difference = 0, min_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %1, %expected, max_ulp_difference = 0, min_ulp_difference = 0 : tensor<f64>, tensor<f64>
    return
  }
}

// Both derivatives keep the forward loop and the reverse loop, each with its
// lazy branch and the branch nested in it, and no enzyme.get/set/push/pop
// survives. The derivative's value is checked by the interpreter run above.
// CHECK-NOT:       enzyme.{{get|set|push|pop|init}}
// CHECK-LABEL:   func.func private @diffenested_if(
// CHECK:           stablehlo.while
// CHECK:           "stablehlo.if"
// CHECK:           "stablehlo.if"
// CHECK:           stablehlo.while
// CHECK:           "stablehlo.if"
// CHECK:           "stablehlo.if"
// CHECK:           return
// CHECK-NOT:       enzyme.{{get|set|push|pop|init}}
// CHECK-LABEL:   func.func private @diffenested_case(
// CHECK:           stablehlo.while
// CHECK:           "stablehlo.if"
// CHECK:           "stablehlo.case"
// CHECK:           stablehlo.while
// CHECK:           "stablehlo.if"
// CHECK:           "stablehlo.case"
// CHECK:           return
// CHECK-NOT:       enzyme.{{get|set|push|pop|init}}
