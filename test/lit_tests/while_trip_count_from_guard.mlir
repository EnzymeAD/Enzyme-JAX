// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=while_trip_count_from_guard" --transform-interpreter --enzyme-hlo-remove-transform --enzyme-hlo-opt --split-input-file | FileCheck %s

// The widened inner loop of a triangular nest (durbin's, after #3587): `j`
// runs the whole hull, with the store `y[j+1][i]` guarded by `j < i` and
// writing the slot's own value back otherwise. Iterations from `i` on do
// nothing: iterate to min(1999, i), where the guard always holds and the
// select (with its read of the slot) folds away.
func.func @triangular(%arg0: tensor<2000x2000xf64>) -> tensor<2000x2000xf64> {
  %c = stablehlo.constant dense<-1> : tensor<i64>
  %c_0 = stablehlo.constant dense<1999> : tensor<i64>
  %c_1 = stablehlo.constant dense<0> : tensor<i64>
  %c_2 = stablehlo.constant dense<1> : tensor<i64>
  %c_3 = stablehlo.constant dense<2000> : tensor<i64>
  %0:2 = stablehlo.while(%iterArg = %c_2, %iterArg_4 = %arg0) : tensor<i64>, tensor<2000x2000xf64>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1:2 = stablehlo.while(%iterArg_5 = %c_1, %iterArg_6 = %iterArg_4) : tensor<i64>, tensor<2000x2000xf64>
    cond {
      %3 = stablehlo.compare LT, %iterArg_5, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3 = stablehlo.subtract %iterArg, %iterArg_5 : tensor<i64>
      %4 = stablehlo.add %3, %c : tensor<i64>
      %5 = stablehlo.compare GE, %4, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %6 = stablehlo.dynamic_slice %iterArg_6, %iterArg_5, %iterArg, sizes = [1, 1] : (tensor<2000x2000xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
      %7 = stablehlo.add %iterArg, %c : tensor<i64>
      %8 = stablehlo.dynamic_slice %iterArg_6, %iterArg_5, %7, sizes = [1, 1] : (tensor<2000x2000xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
      %9 = stablehlo.add %6, %8 : tensor<1x1xf64>
      %10 = stablehlo.add %iterArg_5, %c_2 : tensor<i64>
      %11 = stablehlo.dynamic_slice %iterArg_6, %10, %iterArg, sizes = [1, 1] : (tensor<2000x2000xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
      %12 = stablehlo.select %5, %9, %11 : tensor<i1>, tensor<1x1xf64>
      %13 = stablehlo.dynamic_update_slice %iterArg_6, %12, %10, %iterArg : (tensor<2000x2000xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<2000x2000xf64>
      stablehlo.return %10, %13 : tensor<i64>, tensor<2000x2000xf64>
    }
    %2 = stablehlo.add %iterArg, %c_2 : tensor<i64>
    stablehlo.return %2, %1#1 : tensor<i64>, tensor<2000x2000xf64>
  }
  return %0#1 : tensor<2000x2000xf64>
}

// CHECK-LABEL: func.func @triangular(
// CHECK: stablehlo.while(%[[I:.+]] = %{{.+}}, %[[OUTER:.+]] = %arg0)
// CHECK: } do {
// CHECK: %[[LIM:.+]] = stablehlo.minimum %{{.+}}, %[[I]] {{.*}}: tensor<i64>
// CHECK: stablehlo.while(%[[J:.+]] = %{{.+}}, %[[BUF:.+]] = %[[OUTER]])
// CHECK-NEXT: cond {
// CHECK-NEXT: stablehlo.compare LT, %[[J]], %[[LIM]]
// CHECK: } do {
// CHECK-NOT: stablehlo.select
// CHECK: %[[NEW:.+]] = stablehlo.add %{{.+}}, %{{.+}} : tensor<1x1xf64>
// CHECK-NOT: stablehlo.select
// CHECK: stablehlo.dynamic_update_slice %[[BUF]], %[[NEW]]

// -----

// A scalar carry guarded by `j <= n`, the guard broadcast and the arms
// swapped behind a `not`: iterate to n + 1.
func.func @scalar(%x: tensor<64xf64>, %n: tensor<i64>) -> tensor<f64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c64 = stablehlo.constant dense<64> : tensor<i64>
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %0:2 = stablehlo.while(%j = %c0, %acc = %zero) : tensor<i64>, tensor<f64>
  cond {
    %1 = stablehlo.compare LT, %j, %c64 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.compare LE, %j, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %2 = stablehlo.not %1 : tensor<i1>
    %3 = stablehlo.dynamic_slice %x, %j, sizes = [1] : (tensor<64xf64>, tensor<i64>) -> tensor<1xf64>
    %4 = stablehlo.reshape %3 : (tensor<1xf64>) -> tensor<f64>
    %5 = stablehlo.add %acc, %4 : tensor<f64>
    %6 = stablehlo.select %2, %acc, %5 : tensor<i1>, tensor<f64>
    %7 = stablehlo.add %j, %c1 : tensor<i64>
    stablehlo.return %7, %6 : tensor<i64>, tensor<f64>
  }
  return %0#1 : tensor<f64>
}

// CHECK-LABEL: func.func @scalar(
// CHECK-SAME: %{{.+}}: tensor<64xf64>, %[[N:.+]]: tensor<i64>
// CHECK: %[[B:.+]] = stablehlo.add %[[N]], %{{.+}} {{.*}}: tensor<i64>
// CHECK: %[[LIM:.+]] = stablehlo.minimum %{{.+}}, %[[B]] {{.*}}: tensor<i64>
// CHECK: stablehlo.while(%[[J:.+]] = %{{.+}}, %[[ACC:.+]] = %{{.+}})
// CHECK-NEXT: cond {
// CHECK-NEXT: stablehlo.compare LT, %[[J]], %[[LIM]]
// CHECK: } do {
// CHECK-NOT: stablehlo.select
// CHECK: %[[S:.+]] = stablehlo.add %[[ACC]], %{{.+}} : tensor<f64>
// CHECK-NOT: stablehlo.select
// CHECK: stablehlo.return %{{.+}}, %[[S]]

// -----

// Not every carried value is guarded: the second one accumulates regardless,
// so the late iterations are not no-ops. Left alone.
func.func @unguarded(%x: tensor<64xf64>, %n: tensor<i64>) -> (tensor<f64>, tensor<f64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c64 = stablehlo.constant dense<64> : tensor<i64>
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %0:3 = stablehlo.while(%j = %c0, %acc = %zero, %all = %zero) : tensor<i64>, tensor<f64>, tensor<f64>
  cond {
    %1 = stablehlo.compare LT, %j, %c64 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.compare LT, %j, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %3 = stablehlo.dynamic_slice %x, %j, sizes = [1] : (tensor<64xf64>, tensor<i64>) -> tensor<1xf64>
    %4 = stablehlo.reshape %3 : (tensor<1xf64>) -> tensor<f64>
    %5 = stablehlo.add %acc, %4 : tensor<f64>
    %6 = stablehlo.select %1, %5, %acc : tensor<i1>, tensor<f64>
    %8 = stablehlo.add %all, %4 : tensor<f64>
    %7 = stablehlo.add %j, %c1 : tensor<i64>
    stablehlo.return %7, %6, %8 : tensor<i64>, tensor<f64>, tensor<f64>
  }
  return %0#1, %0#2 : tensor<f64>, tensor<f64>
}

// CHECK-LABEL: func.func @unguarded(
// CHECK-NOT: stablehlo.minimum
// CHECK: stablehlo.select

// -----

// A guard that is a lower bound, `j >= n`, is false first and true after:
// the no-op iterations are a prefix, which shortening the limit cannot
// remove. Left alone.
func.func @lower(%x: tensor<64xf64>, %n: tensor<i64>) -> tensor<f64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c64 = stablehlo.constant dense<64> : tensor<i64>
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %0:2 = stablehlo.while(%j = %c0, %acc = %zero) : tensor<i64>, tensor<f64>
  cond {
    %1 = stablehlo.compare LT, %j, %c64 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.compare GE, %j, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %3 = stablehlo.dynamic_slice %x, %j, sizes = [1] : (tensor<64xf64>, tensor<i64>) -> tensor<1xf64>
    %4 = stablehlo.reshape %3 : (tensor<1xf64>) -> tensor<f64>
    %5 = stablehlo.add %acc, %4 : tensor<f64>
    %6 = stablehlo.select %1, %5, %acc : tensor<i1>, tensor<f64>
    %7 = stablehlo.add %j, %c1 : tensor<i64>
    stablehlo.return %7, %6 : tensor<i64>, tensor<f64>
  }
  return %0#1 : tensor<f64>
}

// CHECK-LABEL: func.func @lower(
// CHECK-NOT: stablehlo.minimum
// CHECK: stablehlo.select
