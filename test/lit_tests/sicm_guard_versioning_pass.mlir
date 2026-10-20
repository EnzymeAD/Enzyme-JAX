// RUN: enzymexlamlir-opt %s --split-input-file --sicm-guard-versioning | FileCheck %s

func.func @loop(%v: tensor<3xf64>, %x0: tensor<3xf64>, %n: tensor<i64>) -> tensor<3xf64> {
  %inf = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %ones = stablehlo.constant dense<1.0> : tensor<3xf64>
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %min = stablehlo.reduce(%v init: %inf) applies stablehlo.minimum across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
  %ok = stablehlo.compare GT, %min, %zero : (tensor<f64>, tensor<f64>) -> tensor<i1>
  %safe = stablehlo.select %ok, %v, %ones : tensor<i1>, tensor<3xf64>
  %setup = stablehlo.sqrt %safe : tensor<3xf64>
  %unrelated = stablehlo.exponential %x0 : tensor<3xf64>
  %r:2 = stablehlo.while(%i = %c0, %x = %unrelated) : tensor<i64>, tensor<3xf64>
  cond {
    %lt = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %lt : tensor<i1>
  } do {
    %y = "stablehlo.if"(%ok) ({
      %a = stablehlo.multiply %x, %setup : tensor<3xf64>
      stablehlo.return %a : tensor<3xf64>
    }, {
      %b = stablehlo.add %x, %v : tensor<3xf64>
      stablehlo.return %b : tensor<3xf64>
    }) {sicm.runtime_guard} : (tensor<i1>) -> tensor<3xf64>
    %i1 = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %i1, %y : tensor<i64>, tensor<3xf64>
  }
  %out = stablehlo.negate %r#1 : tensor<3xf64>
  return %out : tensor<3xf64>
}

// CHECK-LABEL: func.func @loop
// CHECK-SAME: (%[[V:[^:]+]]: tensor<3xf64>, %[[X0:[^:]+]]: tensor<3xf64>, %[[N:[^:]+]]: tensor<i64>)
// CHECK: %[[MIN:.+]] = stablehlo.reduce(%[[V]] init: %{{.+}}) applies stablehlo.minimum across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[OK:.+]] = stablehlo.compare GT, %[[MIN]], %{{.+}} : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CHECK-NEXT: %[[U:.+]] = stablehlo.exponential %[[X0]] : tensor<3xf64>
// CHECK-NEXT: %[[R:.+]] = "stablehlo.if"(%[[OK]]) ({
// CHECK-NEXT: %[[SQ:.+]] = stablehlo.sqrt %[[V]] : tensor<3xf64>
// CHECK-NEXT: %[[W1:.+]]:2 = stablehlo.while(%[[I1:.+]] = %{{.+}}, %[[XA:.+]] = %[[U]]) : tensor<i64>, tensor<3xf64>
// CHECK: } do {
// CHECK-NEXT: %[[A:.+]] = stablehlo.multiply %[[XA]], %[[SQ]] : tensor<3xf64>
// CHECK-NEXT: %[[I1N:.+]] = stablehlo.add %[[I1]], %{{.+}} : tensor<i64>
// CHECK-NEXT: stablehlo.return %[[I1N]], %[[A]] : tensor<i64>, tensor<3xf64>
// CHECK-NEXT: }
// CHECK-NEXT: stablehlo.return %[[W1]]#1 : tensor<3xf64>
// CHECK-NEXT: }, {
// CHECK-NEXT: %[[W2:.+]]:2 = stablehlo.while(%[[I2:.+]] = %{{.+}}, %[[XB:.+]] = %[[U]]) : tensor<i64>, tensor<3xf64>
// CHECK: } do {
// CHECK-NEXT: %[[B:.+]] = stablehlo.add %[[XB]], %[[V]] : tensor<3xf64>
// CHECK-NEXT: %[[I2N:.+]] = stablehlo.add %[[I2]], %{{.+}} : tensor<i64>
// CHECK-NEXT: stablehlo.return %[[I2N]], %[[B]] : tensor<i64>, tensor<3xf64>
// CHECK-NEXT: }
// CHECK-NEXT: stablehlo.return %[[W2]]#1 : tensor<3xf64>
// CHECK-NEXT: }) : (tensor<i1>) -> tensor<3xf64>
// CHECK-NEXT: %[[OUT:.+]] = stablehlo.negate %[[R]] : tensor<3xf64>
// CHECK-NEXT: return %[[OUT]] : tensor<3xf64>

// -----

// Check not hoisted: nothing to version; the conditional loses its marker.
func.func @not_hoisted(%v: tensor<3xf64>, %x: tensor<3xf64>) -> tensor<3xf64> {
  %inf = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %min = stablehlo.reduce(%v init: %inf) applies stablehlo.minimum across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
  %ok = stablehlo.compare GT, %min, %zero : (tensor<f64>, tensor<f64>) -> tensor<i1>
  %y = "stablehlo.if"(%ok) ({
    %a = stablehlo.multiply %x, %v : tensor<3xf64>
    stablehlo.return %a : tensor<3xf64>
  }, {
    %b = stablehlo.add %x, %v : tensor<3xf64>
    stablehlo.return %b : tensor<3xf64>
  }) {sicm.runtime_guard} : (tensor<i1>) -> tensor<3xf64>
  return %y : tensor<3xf64>
}

// CHECK-LABEL: func.func @not_hoisted
// CHECK: %[[OK:.+]] = stablehlo.compare GT
// CHECK-NEXT: %{{.+}} = "stablehlo.if"(%[[OK]]) ({
// CHECK: }) : (tensor<i1>) -> tensor<3xf64>
// CHECK-NOT: sicm.runtime_guard
