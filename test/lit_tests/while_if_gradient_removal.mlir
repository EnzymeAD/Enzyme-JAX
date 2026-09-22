// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --canonicalize --cse | FileCheck %s

// Reverse mode of a counted stablehlo.while whose body updates the accumulator
// under a stablehlo.if -- the shape Reactant emits for a traced loop with a
// traced conditional:
//
//   total = 0
//   for i in 0..N: if i > 0: total += q[i]
//
// remove-unnecessary-enzyme-ops hands the reverse `stablehlo.if` to
// IfOpEnzymeOpsRemover before the reverse `stablehlo.while` around it. The if
// remover reroutes the branch's enzyme.get/enzyme.set through its results, but
// used to leave the original ops behind inside the branch; the while remover
// then found an enzyme.set that was not a direct child of its body and the
// whole pipeline failed with
//   "had set op which was not a direct descendant".
// The loop and the lazy branch must both be retained: no data-length
// unrolling and no eager evaluation of both branches.

module {
  // Reactant marks its traced loops enzyme.disable_mincut; that is the shape
  // that failed, so it is the primary case.
  func.func private @cond_sum(%q: tensor<6xf64>) -> tensor<f64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c6 = stablehlo.constant dense<6> : tensor<i64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %r:2 = stablehlo.while(%i = %c0, %acc = %zero) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut}
    cond {
      %lt = stablehlo.compare LT, %i, %c6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %lt : tensor<i1>
    } do {
      %pred = stablehlo.compare GT, %i, %c0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %next = "stablehlo.if"(%pred) ({
        %e = stablehlo.dynamic_slice %q, %i, sizes = [1] : (tensor<6xf64>, tensor<i64>) -> tensor<1xf64>
        %s = stablehlo.reshape %e : (tensor<1xf64>) -> tensor<f64>
        %a = stablehlo.add %acc, %s : tensor<f64>
        stablehlo.return %a : tensor<f64>
      }, {
        stablehlo.return %acc : tensor<f64>
      }) : (tensor<i1>) -> tensor<f64>
      %inext = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %inext, %next : tensor<i64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }

  // Same loop with the min cut left on.
  func.func private @cond_sum_mincut(%q: tensor<6xf64>) -> tensor<f64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c6 = stablehlo.constant dense<6> : tensor<i64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %r:2 = stablehlo.while(%i = %c0, %acc = %zero) : tensor<i64>, tensor<f64>
    cond {
      %lt = stablehlo.compare LT, %i, %c6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %lt : tensor<i1>
    } do {
      %pred = stablehlo.compare GT, %i, %c0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %next = "stablehlo.if"(%pred) ({
        %e = stablehlo.dynamic_slice %q, %i, sizes = [1] : (tensor<6xf64>, tensor<i64>) -> tensor<1xf64>
        %s = stablehlo.reshape %e : (tensor<1xf64>) -> tensor<f64>
        %a = stablehlo.add %acc, %s : tensor<f64>
        stablehlo.return %a : tensor<f64>
      }, {
        stablehlo.return %acc : tensor<f64>
      }) : (tensor<i1>) -> tensor<f64>
      %inext = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %inext, %next : tensor<i64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }

  // Control: the three-element loop (kept as a loop here; as a Julia program it
  // only passed because it was unrolled before differentiation).
  func.func private @cond_sum3(%q: tensor<3xf64>) -> tensor<f64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c3 = stablehlo.constant dense<3> : tensor<i64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %r:2 = stablehlo.while(%i = %c0, %acc = %zero) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut}
    cond {
      %lt = stablehlo.compare LT, %i, %c3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %lt : tensor<i1>
    } do {
      %pred = stablehlo.compare GT, %i, %c0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %next = "stablehlo.if"(%pred) ({
        %e = stablehlo.dynamic_slice %q, %i, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> tensor<1xf64>
        %s = stablehlo.reshape %e : (tensor<1xf64>) -> tensor<f64>
        %a = stablehlo.add %acc, %s : tensor<f64>
        stablehlo.return %a : tensor<f64>
      }, {
        stablehlo.return %acc : tensor<f64>
      }) : (tensor<i1>) -> tensor<f64>
      %inext = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %inext, %next : tensor<i64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }

  // Control: the branch-free loop that always worked.
  func.func private @sum_all(%q: tensor<6xf64>) -> tensor<f64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c6 = stablehlo.constant dense<6> : tensor<i64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %r:2 = stablehlo.while(%i = %c0, %acc = %zero) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut}
    cond {
      %lt = stablehlo.compare LT, %i, %c6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %lt : tensor<i1>
    } do {
      %e = stablehlo.dynamic_slice %q, %i, sizes = [1] : (tensor<6xf64>, tensor<i64>) -> tensor<1xf64>
      %s = stablehlo.reshape %e : (tensor<1xf64>) -> tensor<f64>
      %a = stablehlo.add %acc, %s : tensor<f64>
      %inext = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %inext, %a : tensor<i64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }

  func.func @main(%q: tensor<6xf64>, %q3: tensor<3xf64>) -> (tensor<6xf64>, tensor<6xf64>, tensor<3xf64>, tensor<6xf64>) {
    %seed = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0 = enzyme.autodiff @cond_sum(%q, %seed) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<6xf64>, tensor<f64>) -> tensor<6xf64>
    %1 = enzyme.autodiff @cond_sum_mincut(%q, %seed) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<6xf64>, tensor<f64>) -> tensor<6xf64>
    %2 = enzyme.autodiff @cond_sum3(%q3, %seed) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<3xf64>, tensor<f64>) -> tensor<3xf64>
    %3 = enzyme.autodiff @sum_all(%q, %seed) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<6xf64>, tensor<f64>) -> tensor<6xf64>
    return %0, %1, %2, %3 : tensor<6xf64>, tensor<6xf64>, tensor<3xf64>, tensor<6xf64>
  }
}

// Every derivative keeps its reverse loop with the lazy branch inside it, the
// variants that tape keep the forward loop too, and no enzyme.get/set/push/pop
// survives anywhere in the module.
// CHECK-NOT:       enzyme.{{get|set|push|pop|init}}
// CHECK-LABEL:   func.func private @diffecond_sum(
// CHECK:           stablehlo.while
// CHECK:           stablehlo.if
// CHECK:           stablehlo.while
// CHECK:           stablehlo.if
// CHECK:           return
// CHECK-NOT:       enzyme.{{get|set|push|pop|init}}
// With the min cut on nothing is taped for this add-only loop: the branch
// predicate and the slice index are recomputed from the reverse counter, and
// the primal is not returned, so the augmented forward loop is dead and only
// the reverse loop with its lazy branch remains.
// CHECK-LABEL:   func.func private @diffecond_sum_mincut(
// CHECK:           stablehlo.while
// CHECK:           stablehlo.if
// CHECK:           return
// CHECK-NOT:       enzyme.{{get|set|push|pop|init}}
// CHECK-LABEL:   func.func private @diffecond_sum3(
// CHECK:           stablehlo.while
// CHECK:           stablehlo.if
// CHECK:           stablehlo.while
// CHECK:           stablehlo.if
// CHECK:           return
// CHECK-NOT:       enzyme.{{get|set|push|pop|init}}
// CHECK-LABEL:   func.func private @diffesum_all(
// CHECK:           stablehlo.while
// CHECK:           stablehlo.while
// CHECK:           return
// CHECK-NOT:       enzyme.{{get|set|push|pop|init}}
