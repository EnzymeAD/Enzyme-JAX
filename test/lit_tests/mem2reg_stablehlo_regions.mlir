// RUN: enzymexlamlir-opt %s --mem2reg --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @while_store
// CHECK-SAME: (%[[N:.+]]: tensor<i64>, %[[X:.+]]: !stablehlo.token)
// CHECK-NOT: enzyme.
// CHECK: %[[W:.+]]:2 = stablehlo.while(%[[I:[^ ]+]] = %{{[^ ]+}}, %[[ACC:[^ ]+]] = %[[X]])
// CHECK: stablehlo.compare
// CHECK: } do {
// CHECK: %[[SUM:.+]] = stablehlo.after_all %[[ACC]], %[[ACC]]
// CHECK: stablehlo.return %{{.+}}, %[[SUM]]
// CHECK: return %[[W]]#1
func.func @while_store(%n: tensor<i64>, %x: !stablehlo.token) -> !stablehlo.token {
  %g = "enzyme.init"() : () -> !enzyme.Gradient<!stablehlo.token>
  "enzyme.set"(%g, %x) : (!enzyme.Gradient<!stablehlo.token>, !stablehlo.token) -> ()
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %w = stablehlo.while(%i = %c0) : tensor<i64>
  cond {
    %cmp = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %v = "enzyme.get"(%g) : (!enzyme.Gradient<!stablehlo.token>) -> !stablehlo.token
    %s = stablehlo.after_all %v, %v : !stablehlo.token
    "enzyme.set"(%g, %s) : (!enzyme.Gradient<!stablehlo.token>, !stablehlo.token) -> ()
    %next = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %next : tensor<i64>
  }
  %r = "enzyme.get"(%g) : (!enzyme.Gradient<!stablehlo.token>) -> !stablehlo.token
  return %r : !stablehlo.token
}

// -----

// CHECK-LABEL: func.func @while_load
// CHECK-SAME: (%[[N:.+]]: tensor<i64>, %[[X:.+]]: !stablehlo.token, %[[T:.+]]: !stablehlo.token)
// CHECK-NOT: enzyme.
// CHECK: stablehlo.while(%{{.+}} = %{{.+}}, %{{.+}} = %[[T]]) : tensor<i64>, !stablehlo.token
// CHECK: } do {
// CHECK: stablehlo.after_all %{{.+}}, %[[X]]
func.func @while_load(%n: tensor<i64>, %x: !stablehlo.token, %t: !stablehlo.token) -> !stablehlo.token {
  %g = "enzyme.init"() : () -> !enzyme.Gradient<!stablehlo.token>
  "enzyme.set"(%g, %x) : (!enzyme.Gradient<!stablehlo.token>, !stablehlo.token) -> ()
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %w:2 = stablehlo.while(%i = %c0, %acc = %t) : tensor<i64>, !stablehlo.token
  cond {
    %cmp = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %v = "enzyme.get"(%g) : (!enzyme.Gradient<!stablehlo.token>) -> !stablehlo.token
    %s = stablehlo.after_all %acc, %v : !stablehlo.token
    stablehlo.return %i, %s : tensor<i64>, !stablehlo.token
  }
  return %w#1 : !stablehlo.token
}

// -----

// CHECK-LABEL: func.func @while_cond_store
// CHECK: enzyme.init
// CHECK: enzyme.set
func.func @while_cond_store(%n: tensor<i64>, %x: !stablehlo.token) -> !stablehlo.token {
  %g = "enzyme.init"() : () -> !enzyme.Gradient<!stablehlo.token>
  "enzyme.set"(%g, %x) : (!enzyme.Gradient<!stablehlo.token>, !stablehlo.token) -> ()
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %w = stablehlo.while(%i = %c0) : tensor<i64>
  cond {
    %v = "enzyme.get"(%g) : (!enzyme.Gradient<!stablehlo.token>) -> !stablehlo.token
    %s = stablehlo.after_all %v, %v : !stablehlo.token
    "enzyme.set"(%g, %s) : (!enzyme.Gradient<!stablehlo.token>, !stablehlo.token) -> ()
    %cmp = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %next = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %next : tensor<i64>
  }
  %r = "enzyme.get"(%g) : (!enzyme.Gradient<!stablehlo.token>) -> !stablehlo.token
  return %r : !stablehlo.token
}

// -----

// CHECK-LABEL: func.func @if_store
// CHECK-SAME: (%[[P:.+]]: tensor<i1>, %[[X:.+]]: !stablehlo.token)
// CHECK-NOT: enzyme.
// CHECK: %[[IF:.+]] = "stablehlo.if"(%[[P]])
// CHECK: %[[SUM:.+]] = stablehlo.after_all %[[X]], %[[X]]
// CHECK: stablehlo.return %[[SUM]]
// CHECK: }, {
// CHECK: stablehlo.return %[[X]]
// CHECK: }) : (tensor<i1>) -> !stablehlo.token
// CHECK: return %[[IF]]
func.func @if_store(%p: tensor<i1>, %x: !stablehlo.token) -> !stablehlo.token {
  %g = "enzyme.init"() : () -> !enzyme.Gradient<!stablehlo.token>
  "enzyme.set"(%g, %x) : (!enzyme.Gradient<!stablehlo.token>, !stablehlo.token) -> ()
  "stablehlo.if"(%p) ({
    %v = "enzyme.get"(%g) : (!enzyme.Gradient<!stablehlo.token>) -> !stablehlo.token
    %s = stablehlo.after_all %v, %v : !stablehlo.token
    "enzyme.set"(%g, %s) : (!enzyme.Gradient<!stablehlo.token>, !stablehlo.token) -> ()
    stablehlo.return
  }, {
    stablehlo.return
  }) : (tensor<i1>) -> ()
  %r = "enzyme.get"(%g) : (!enzyme.Gradient<!stablehlo.token>) -> !stablehlo.token
  return %r : !stablehlo.token
}

// -----

// CHECK-LABEL: func.func @case_store
// CHECK-SAME: (%[[IDX:.+]]: tensor<i32>, %[[X:.+]]: !stablehlo.token)
// CHECK-NOT: enzyme.
// CHECK: %[[CASE:.+]]:2 = "stablehlo.case"(%[[IDX]])
// CHECK: %[[A:.+]] = stablehlo.after_all %[[X]], %[[X]]
// CHECK: stablehlo.return %{{.+}}, %[[A]]
// CHECK: }, {
// CHECK: stablehlo.return %{{.+}}, %[[X]]
// CHECK: }, {
// CHECK: %[[M:.+]] = stablehlo.after_all %[[X]]
// CHECK: stablehlo.return %{{.+}}, %[[M]]
// CHECK: }) : (tensor<i32>) -> (tensor<i64>, !stablehlo.token)
// CHECK: return %[[CASE]]#1
func.func @case_store(%idx: tensor<i32>, %x: !stablehlo.token) -> !stablehlo.token {
  %g = "enzyme.init"() : () -> !enzyme.Gradient<!stablehlo.token>
  "enzyme.set"(%g, %x) : (!enzyme.Gradient<!stablehlo.token>, !stablehlo.token) -> ()
  %c = "stablehlo.case"(%idx) ({
    %v = "enzyme.get"(%g) : (!enzyme.Gradient<!stablehlo.token>) -> !stablehlo.token
    %s = stablehlo.after_all %v, %v : !stablehlo.token
    "enzyme.set"(%g, %s) : (!enzyme.Gradient<!stablehlo.token>, !stablehlo.token) -> ()
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    stablehlo.return %c0 : tensor<i64>
  }, {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    stablehlo.return %c1 : tensor<i64>
  }, {
    %v = "enzyme.get"(%g) : (!enzyme.Gradient<!stablehlo.token>) -> !stablehlo.token
    %m = stablehlo.after_all %v : !stablehlo.token
    "enzyme.set"(%g, %m) : (!enzyme.Gradient<!stablehlo.token>, !stablehlo.token) -> ()
    %c2 = stablehlo.constant dense<2> : tensor<i64>
    stablehlo.return %c2 : tensor<i64>
  }) : (tensor<i32>) -> tensor<i64>
  %r = "enzyme.get"(%g) : (!enzyme.Gradient<!stablehlo.token>) -> !stablehlo.token
  return %r : !stablehlo.token
}
