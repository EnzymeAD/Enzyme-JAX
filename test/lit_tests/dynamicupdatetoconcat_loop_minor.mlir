// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=dynamic_update_to_concat<1>" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Minor-dim updates under 128 onto a carried while arg stay a DUS chain.

// CHECK-LABEL: func.func @narrow_minor_in_loop
// CHECK: } do {
// CHECK-NEXT: %{{.+}} = stablehlo.add
// CHECK-NEXT: %{{.+}} = stablehlo.dynamic_update_slice %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : (tensor<1x16xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x16xf64>
// CHECK-NOT: stablehlo.concatenate
func.func @narrow_minor_in_loop(%T0: tensor<1x16xf64>, %u: tensor<1x2xf64>, %n: tensor<i64>) -> tensor<1x16xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c13 = stablehlo.constant dense<13> : tensor<i64>
  %r:2 = stablehlo.while(%i = %c0, %T = %T0) : tensor<i64>, tensor<1x16xf64>
  cond {
    %p = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %ip = stablehlo.add %i, %c1 : tensor<i64>
    %T1 = stablehlo.dynamic_update_slice %T, %u, %c0, %c13 : (tensor<1x16xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x16xf64>
    stablehlo.return %ip, %T1 : tensor<i64>, tensor<1x16xf64>
  }
  return %r#1 : tensor<1x16xf64>
}

// CHECK-LABEL: func.func @wide_minor_in_loop
// CHECK: } do {
// CHECK-NEXT: %{{.+}} = stablehlo.add
// CHECK-NEXT: %[[A:.+]] = stablehlo.slice %{{.+}} [0:1, 0:13] : (tensor<1x200xf64>) -> tensor<1x13xf64>
// CHECK-NEXT: %[[B:.+]] = stablehlo.slice %{{.+}} [0:1, 15:200] : (tensor<1x200xf64>) -> tensor<1x185xf64>
// CHECK-NEXT: %{{.+}} = stablehlo.concatenate %[[A]], %{{.+}}, %[[B]], dim = 1 : (tensor<1x13xf64>, tensor<1x2xf64>, tensor<1x185xf64>) -> tensor<1x200xf64>
func.func @wide_minor_in_loop(%T0: tensor<1x200xf64>, %u: tensor<1x2xf64>, %n: tensor<i64>) -> tensor<1x200xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c13 = stablehlo.constant dense<13> : tensor<i64>
  %r:2 = stablehlo.while(%i = %c0, %T = %T0) : tensor<i64>, tensor<1x200xf64>
  cond {
    %p = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %ip = stablehlo.add %i, %c1 : tensor<i64>
    %T1 = stablehlo.dynamic_update_slice %T, %u, %c0, %c13 : (tensor<1x200xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x200xf64>
    stablehlo.return %ip, %T1 : tensor<i64>, tensor<1x200xf64>
  }
  return %r#1 : tensor<1x200xf64>
}

// CHECK-LABEL: func.func @narrow_minor_outside_loop
// CHECK: %[[A:.+]] = stablehlo.slice %{{.+}} [0:1, 0:13] : (tensor<1x16xf64>) -> tensor<1x13xf64>
// CHECK-NEXT: %[[B:.+]] = stablehlo.slice %{{.+}} [0:1, 15:16] : (tensor<1x16xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %{{.+}} = stablehlo.concatenate %[[A]], %{{.+}}, %[[B]], dim = 1 : (tensor<1x13xf64>, tensor<1x2xf64>, tensor<1x1xf64>) -> tensor<1x16xf64>
func.func @narrow_minor_outside_loop(%T: tensor<1x16xf64>, %u: tensor<1x2xf64>) -> tensor<1x16xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c13 = stablehlo.constant dense<13> : tensor<i64>
  %T1 = stablehlo.dynamic_update_slice %T, %u, %c0, %c13 : (tensor<1x16xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x16xf64>
  return %T1 : tensor<1x16xf64>
}

// CHECK-LABEL: func.func @narrow_major_in_loop
// CHECK: } do {
// CHECK-NEXT: %{{.+}} = stablehlo.add
// CHECK-NEXT: %[[A:.+]] = stablehlo.slice %{{.+}} [0:13, 0:4] : (tensor<16x4xf64>) -> tensor<13x4xf64>
// CHECK-NEXT: %[[B:.+]] = stablehlo.slice %{{.+}} [15:16, 0:4] : (tensor<16x4xf64>) -> tensor<1x4xf64>
// CHECK-NEXT: %{{.+}} = stablehlo.concatenate %[[A]], %{{.+}}, %[[B]], dim = 0 : (tensor<13x4xf64>, tensor<2x4xf64>, tensor<1x4xf64>) -> tensor<16x4xf64>
func.func @narrow_major_in_loop(%T0: tensor<16x4xf64>, %u: tensor<2x4xf64>, %n: tensor<i64>) -> tensor<16x4xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c13 = stablehlo.constant dense<13> : tensor<i64>
  %r:2 = stablehlo.while(%i = %c0, %T = %T0) : tensor<i64>, tensor<16x4xf64>
  cond {
    %p = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %ip = stablehlo.add %i, %c1 : tensor<i64>
    %T1 = stablehlo.dynamic_update_slice %T, %u, %c13, %c0 : (tensor<16x4xf64>, tensor<2x4xf64>, tensor<i64>, tensor<i64>) -> tensor<16x4xf64>
    stablehlo.return %ip, %T1 : tensor<i64>, tensor<16x4xf64>
  }
  return %r#1 : tensor<16x4xf64>
}

// CHECK-LABEL: func.func @narrow_minor_in_loop_not_carried
// CHECK: } do {
// CHECK: stablehlo.concatenate
// CHECK-NOT: stablehlo.dynamic_update_slice
func.func @narrow_minor_in_loop_not_carried(%T0: tensor<1x16xf64>, %u: tensor<1x2xf64>, %n: tensor<i64>) -> tensor<1x16xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c13 = stablehlo.constant dense<13> : tensor<i64>
  %r:2 = stablehlo.while(%i = %c0, %T = %T0) : tensor<i64>, tensor<1x16xf64>
  cond {
    %p = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %ip = stablehlo.add %i, %c1 : tensor<i64>
    %e = stablehlo.exponential %T : tensor<1x16xf64>
    %T1 = stablehlo.dynamic_update_slice %e, %u, %c0, %c13 : (tensor<1x16xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x16xf64>
    stablehlo.return %ip, %T1 : tensor<i64>, tensor<1x16xf64>
  }
  return %r#1 : tensor<1x16xf64>
}
