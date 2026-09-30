// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=while_invariant_concat_range" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Zero iterations keep the initial range, hence select(ran, K, init[2:13]).

// CHECK-LABEL: func.func @drop_range
// CHECK-SAME: (%[[T0:.+]]: tensor<1x16xf64>, %[[K:.+]]: tensor<1x11xf64>, %[[N:.+]]: tensor<i64>, %[[V:.+]]: tensor<1x2xf64>)
// CHECK-DAG: %[[C0:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK: %[[LO0:.+]] = stablehlo.slice %[[T0]] [0:1, 0:2] : (tensor<1x16xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[HI0:.+]] = stablehlo.slice %[[T0]] [0:1, 13:16] : (tensor<1x16xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[INIT:.+]] = stablehlo.concatenate %[[LO0]], %[[HI0]], dim = 1 : (tensor<1x2xf64>, tensor<1x3xf64>) -> tensor<1x5xf64>
// CHECK-NEXT: %[[RAN:.+]] = stablehlo.compare LT, %[[C0]], %[[N]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT: %[[MID0:.+]] = stablehlo.slice %[[T0]] [0:1, 2:13] : (tensor<1x16xf64>) -> tensor<1x11xf64>
// CHECK-NEXT: %[[W:.+]]:2 = stablehlo.while(%{{.+}} = %[[C0]], %[[T:.+]] = %[[INIT]]) : tensor<i64>, tensor<1x5xf64>
// CHECK: } do {
// CHECK-NEXT: %{{.+}} = stablehlo.add
// CHECK-NEXT: %[[HD:.+]] = stablehlo.slice %[[T]] [0:1, 0:2] : (tensor<1x5xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[NH:.+]] = stablehlo.add %[[HD]], %[[V]] : tensor<1x2xf64>
// CHECK-NEXT: %[[TL:.+]] = stablehlo.slice %[[T]] [0:1, 2:5] : (tensor<1x5xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[NT:.+]] = stablehlo.multiply %[[TL]], %[[TL]] : tensor<1x3xf64>
// CHECK-NEXT: %[[Y:.+]] = stablehlo.concatenate %[[NH]], %[[NT]], dim = 1 : (tensor<1x2xf64>, tensor<1x3xf64>) -> tensor<1x5xf64>
// CHECK-NEXT: stablehlo.return %{{.+}}, %[[Y]] : tensor<i64>, tensor<1x5xf64>
// CHECK: %[[MID:.+]] = stablehlo.select %[[RAN]], %[[K]], %[[MID0]] : tensor<i1>, tensor<1x11xf64>
// CHECK-NEXT: %[[LO:.+]] = stablehlo.slice %[[W]]#1 [0:1, 0:2] : (tensor<1x5xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[HI:.+]] = stablehlo.slice %[[W]]#1 [0:1, 2:5] : (tensor<1x5xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[R:.+]] = stablehlo.concatenate %[[LO]], %[[MID]], %[[HI]], dim = 1 : (tensor<1x2xf64>, tensor<1x11xf64>, tensor<1x3xf64>) -> tensor<1x16xf64>
// CHECK-NEXT: return %[[R]] : tensor<1x16xf64>
func.func @drop_range(%T0: tensor<1x16xf64>, %K: tensor<1x11xf64>, %n: tensor<i64>, %v: tensor<1x2xf64>) -> tensor<1x16xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %r:2 = stablehlo.while(%i = %c0, %T = %T0) : tensor<i64>, tensor<1x16xf64>
  cond {
    %p = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %ip = stablehlo.add %i, %c1 : tensor<i64>
    %head = stablehlo.slice %T [0:1, 0:2] : (tensor<1x16xf64>) -> tensor<1x2xf64>
    %nh = stablehlo.add %head, %v : tensor<1x2xf64>
    %tail = stablehlo.slice %T [0:1, 13:16] : (tensor<1x16xf64>) -> tensor<1x3xf64>
    %nt = stablehlo.multiply %tail, %tail : tensor<1x3xf64>
    %new = stablehlo.concatenate %nh, %K, %nt, dim = 1 : (tensor<1x2xf64>, tensor<1x11xf64>, tensor<1x3xf64>) -> tensor<1x16xf64>
    stablehlo.return %ip, %new : tensor<i64>, tensor<1x16xf64>
  }
  return %r#1 : tensor<1x16xf64>
}

// CHECK-LABEL: func.func @neg_read_inside
// CHECK: stablehlo.while(%{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}) : tensor<i64>, tensor<1x16xf64>
func.func @neg_read_inside(%T0: tensor<1x16xf64>, %K: tensor<1x11xf64>, %n: tensor<i64>) -> tensor<1x16xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %r:2 = stablehlo.while(%i = %c0, %T = %T0) : tensor<i64>, tensor<1x16xf64>
  cond {
    %p = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %ip = stablehlo.add %i, %c1 : tensor<i64>
    %in = stablehlo.slice %T [0:1, 5:7] : (tensor<1x16xf64>) -> tensor<1x2xf64>
    %tail = stablehlo.slice %T [0:1, 13:16] : (tensor<1x16xf64>) -> tensor<1x3xf64>
    %new = stablehlo.concatenate %in, %K, %tail, dim = 1 : (tensor<1x2xf64>, tensor<1x11xf64>, tensor<1x3xf64>) -> tensor<1x16xf64>
    stablehlo.return %ip, %new : tensor<i64>, tensor<1x16xf64>
  }
  return %r#1 : tensor<1x16xf64>
}
