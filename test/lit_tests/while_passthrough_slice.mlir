// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=while_passthrough_slice" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt="licm_passes=true" | FileCheck %s --check-prefix=OPT

// CHECK-LABEL: func.func @dus_chain
// CHECK-SAME: (%[[T0:.+]]: tensor<1x16xf64>, %[[X:.+]]: tensor<11x2xf64>,
// CHECK: %[[YS:.+]] = stablehlo.slice %[[T0]] [0:1, 2:13] : (tensor<1x16xf64>) -> tensor<1x11xf64>
// CHECK-NEXT: %{{.+}}:2 = stablehlo.while(%{{.+}} = %{{.+}}, %{{.+}} = %[[T0]]) : tensor<i64>, tensor<1x16xf64>
// CHECK: } do {
// CHECK-NEXT: %{{.+}} = stablehlo.add
// CHECK-NEXT: %[[Y:.+]] = stablehlo.reshape %[[YS]] : (tensor<1x11xf64>) -> tensor<11xf64>
// CHECK-NEXT: %{{.+}} = stablehlo.dot_general %[[X]], %[[Y]], contracting_dims = [0] x [0] : (tensor<11x2xf64>, tensor<11xf64>) -> tensor<2xf64>
// The second move slices the intermediate DUS result; left to the OPT run.
// CHECK: stablehlo.slice %{{.+}} [0:1, 2:13] : (tensor<1x16xf64>) -> tensor<1x11xf64>

// columns leave the loop. The trace (minor dimension 16 < 128) keeps its
// in-place dynamic_update_slice write-back of the value and tail windows; no
// ys-wide op remains in the loop body.
// OPT-LABEL: func.func @dus_chain
// OPT-SAME: (%[[T0:.+]]: tensor<1x16xf64>,
// OPT: %[[YS:.+]] = stablehlo.slice %[[T0]] [0:1, 2:13] : (tensor<1x16xf64>) -> tensor<1x11xf64>
// OPT: stablehlo.dot_general
// OPT: stablehlo.dot_general
// OPT: %{{.+}}:2 = stablehlo.while(%{{.+}} = %{{.+}}, %{{.+}} = %[[T0]]) : tensor<i64>, tensor<1x16xf64>
// OPT-NOT: stablehlo.dot_general
// OPT-NOT: tensor<1x11xf64>
// OPT: stablehlo.dynamic_update_slice
// OPT-NOT: stablehlo.concatenate
// OPT: return
func.func @dus_chain(%T0: tensor<1x16xf64>, %X: tensor<11x2xf64>, %n: tensor<i64>, %u: tensor<2xf64>) -> tensor<1x16xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c13 = stablehlo.constant dense<13> : tensor<i64>
  %half = stablehlo.constant dense<0.5> : tensor<f64>
  %r:2 = stablehlo.while(%i = %c0, %T = %T0) : tensor<i64>, tensor<1x16xf64>
  cond {
    %p = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %ip = stablehlo.add %i, %c1 : tensor<i64>
    %ys = stablehlo.slice %T [0:1, 2:13] : (tensor<1x16xf64>) -> tensor<1x11xf64>
    %y = stablehlo.reshape %ys : (tensor<1x11xf64>) -> tensor<11xf64>
    %c = stablehlo.dot_general %X, %y, contracting_dims = [0] x [0] : (tensor<11x2xf64>, tensor<11xf64>) -> tensor<2xf64>
    %c0v = stablehlo.slice %c [0:1] : (tensor<2xf64>) -> tensor<1xf64>
    %s = stablehlo.reshape %c0v : (tensor<1xf64>) -> tensor<1x1xf64>
    %uu = stablehlo.slice %u [0:1] : (tensor<2xf64>) -> tensor<1xf64>
    %us = stablehlo.reshape %uu : (tensor<1xf64>) -> tensor<f64>
    %acc = stablehlo.compare LT, %us, %half : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %old0 = stablehlo.slice %T [0:1, 0:1] : (tensor<1x16xf64>) -> tensor<1x1xf64>
    %v0 = stablehlo.select %acc, %s, %old0 : tensor<i1>, tensor<1x1xf64>
    %T1 = stablehlo.dynamic_update_slice %T, %v0, %c0, %c0 : (tensor<1x16xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x16xf64>
    %tn = stablehlo.broadcast_in_dim %s, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<1x3xf64>
    %oldt = stablehlo.slice %T1 [0:1, 13:16] : (tensor<1x16xf64>) -> tensor<1x3xf64>
    %vt = stablehlo.select %acc, %tn, %oldt : tensor<i1>, tensor<1x3xf64>
    %T2 = stablehlo.dynamic_update_slice %T1, %vt, %c0, %c13 : (tensor<1x16xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x16xf64>
    %ys2 = stablehlo.slice %T2 [0:1, 2:13] : (tensor<1x16xf64>) -> tensor<1x11xf64>
    %y2 = stablehlo.reshape %ys2 : (tensor<1x11xf64>) -> tensor<11xf64>
    %c2 = stablehlo.dot_general %X, %y2, contracting_dims = [0] x [0] : (tensor<11x2xf64>, tensor<11xf64>) -> tensor<2xf64>
    %c1v = stablehlo.slice %c2 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
    %s2 = stablehlo.reshape %c1v : (tensor<1xf64>) -> tensor<1x1xf64>
    %uu2 = stablehlo.slice %u [1:2] : (tensor<2xf64>) -> tensor<1xf64>
    %us2 = stablehlo.reshape %uu2 : (tensor<1xf64>) -> tensor<f64>
    %acc2 = stablehlo.compare LT, %us2, %half : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %old1 = stablehlo.slice %T2 [0:1, 1:2] : (tensor<1x16xf64>) -> tensor<1x1xf64>
    %v1 = stablehlo.select %acc2, %s2, %old1 : tensor<i1>, tensor<1x1xf64>
    %T3 = stablehlo.dynamic_update_slice %T2, %v1, %c0, %c1 : (tensor<1x16xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x16xf64>
    %tn2 = stablehlo.broadcast_in_dim %s2, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<1x3xf64>
    %oldt2 = stablehlo.slice %T3 [0:1, 13:16] : (tensor<1x16xf64>) -> tensor<1x3xf64>
    %vt2 = stablehlo.select %acc2, %tn2, %oldt2 : tensor<i1>, tensor<1x3xf64>
    %T4 = stablehlo.dynamic_update_slice %T3, %vt2, %c0, %c13 : (tensor<1x16xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x16xf64>
    stablehlo.return %ip, %T4 : tensor<i64>, tensor<1x16xf64>
  }
  return %r#1 : tensor<1x16xf64>
}

// CHECK-LABEL: func.func @select_concat
// CHECK-SAME: (%[[T0:.+]]: tensor<1x16xf64>, %[[X:.+]]: tensor<11x2xf64>,
// CHECK: %[[KEEP:.+]] = stablehlo.slice %[[T0]] [0:1, 2:13] : (tensor<1x16xf64>) -> tensor<1x11xf64>
// CHECK-NEXT: %[[TAIL:.+]] = stablehlo.slice %[[T0]] [0:1, 13:16] : (tensor<1x16xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[YS:.+]] = stablehlo.slice %[[T0]] [0:1, 2:13] : (tensor<1x16xf64>) -> tensor<1x11xf64>
// CHECK-NEXT: %{{.+}}:2 = stablehlo.while(%{{.+}} = %{{.+}}, %[[T:.+]] = %[[T0]]) : tensor<i64>, tensor<1x16xf64>
// CHECK: } do {
// CHECK-NEXT: %{{.+}} = stablehlo.add
// CHECK-NEXT: %[[Y:.+]] = stablehlo.reshape %[[YS]] : (tensor<1x11xf64>) -> tensor<11xf64>
// CHECK-NEXT: %[[C:.+]] = stablehlo.dot_general %[[X]], %[[Y]], contracting_dims = [0] x [0] : (tensor<11x2xf64>, tensor<11xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[V:.+]] = stablehlo.reshape %[[C]] : (tensor<2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %{{.+}} = stablehlo.concatenate %[[V]], %[[KEEP]], %[[TAIL]], dim = 1 : (tensor<1x2xf64>, tensor<1x11xf64>, tensor<1x3xf64>) -> tensor<1x16xf64>
func.func @select_concat(%T0: tensor<1x16xf64>, %X: tensor<11x2xf64>, %n: tensor<i64>, %u: tensor<f64>) -> tensor<1x16xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %half = stablehlo.constant dense<0.5> : tensor<f64>
  %r:2 = stablehlo.while(%i = %c0, %T = %T0) : tensor<i64>, tensor<1x16xf64>
  cond {
    %p = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %ip = stablehlo.add %i, %c1 : tensor<i64>
    %ys = stablehlo.slice %T [0:1, 2:13] : (tensor<1x16xf64>) -> tensor<1x11xf64>
    %y = stablehlo.reshape %ys : (tensor<1x11xf64>) -> tensor<11xf64>
    %c = stablehlo.dot_general %X, %y, contracting_dims = [0] x [0] : (tensor<11x2xf64>, tensor<11xf64>) -> tensor<2xf64>
    %v = stablehlo.reshape %c : (tensor<2xf64>) -> tensor<1x2xf64>
    %tail = stablehlo.slice %T [0:1, 13:16] : (tensor<1x16xf64>) -> tensor<1x3xf64>
    %keep = stablehlo.slice %T [0:1, 2:13] : (tensor<1x16xf64>) -> tensor<1x11xf64>
    %new = stablehlo.concatenate %v, %keep, %tail, dim = 1 : (tensor<1x2xf64>, tensor<1x11xf64>, tensor<1x3xf64>) -> tensor<1x16xf64>
    %acc = stablehlo.compare LT, %u, %half : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %T1 = stablehlo.select %acc, %new, %T : tensor<i1>, tensor<1x16xf64>
    stablehlo.return %ip, %T1 : tensor<i64>, tensor<1x16xf64>
  }
  return %r#1 : tensor<1x16xf64>
}

// CHECK-LABEL: func.func @neg_overlap
// CHECK: stablehlo.while
// CHECK: } do {
// CHECK-NEXT: %{{.+}} = stablehlo.add
// CHECK-NEXT: %{{.+}} = stablehlo.slice %{{.+}} [0:1, 2:13] : (tensor<1x16xf64>) -> tensor<1x11xf64>
func.func @neg_overlap(%T0: tensor<1x16xf64>, %X: tensor<11x2xf64>, %n: tensor<i64>) -> tensor<1x16xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c5 = stablehlo.constant dense<5> : tensor<i64>
  %r:2 = stablehlo.while(%i = %c0, %T = %T0) : tensor<i64>, tensor<1x16xf64>
  cond {
    %p = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %ip = stablehlo.add %i, %c1 : tensor<i64>
    %ys = stablehlo.slice %T [0:1, 2:13] : (tensor<1x16xf64>) -> tensor<1x11xf64>
    %y = stablehlo.reshape %ys : (tensor<1x11xf64>) -> tensor<11xf64>
    %c = stablehlo.dot_general %X, %y, contracting_dims = [0] x [0] : (tensor<11x2xf64>, tensor<11xf64>) -> tensor<2xf64>
    %c0v = stablehlo.slice %c [0:1] : (tensor<2xf64>) -> tensor<1xf64>
    %s = stablehlo.reshape %c0v : (tensor<1xf64>) -> tensor<1x1xf64>
    %T1 = stablehlo.dynamic_update_slice %T, %s, %c0, %c5 : (tensor<1x16xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x16xf64>
    stablehlo.return %ip, %T1 : tensor<i64>, tensor<1x16xf64>
  }
  return %r#1 : tensor<1x16xf64>
}
