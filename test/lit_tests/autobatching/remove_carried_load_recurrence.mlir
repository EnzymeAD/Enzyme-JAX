// RUN: enzymexlamlir-opt %s '--pass-pipeline=builtin.module(enzyme-hlo-generate-td{patterns=remove_loop_carried_dependencies_from_while_load_operations},transform-interpreter,enzyme-hlo-remove-transform)' --split-input-file | FileCheck %s

// y[j+1][i] = y[j][i] + y[j][i-1]: a recurrence down the rows. The guard's
// "else" read of y[j+1][i] is a self-lane load (iteration j only writes row
// j+1), but the other reads keep the buffer carried, so moving it to the
// loop operand would only keep that operand live across the inner loop --
// a copy of the whole buffer on every entry (issue #3598). Left alone.
func.func @recurrence(%arg0: tensor<2000x2000xf64>) -> tensor<2000x2000xf64> {
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
      %3 = stablehlo.multiply %iterArg_5, %c : tensor<i64>
      %4 = stablehlo.add %iterArg, %3 : tensor<i64>
      %5 = stablehlo.add %4, %c : tensor<i64>
      %6 = stablehlo.compare GE, %5, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %7 = stablehlo.dynamic_slice %iterArg_6, %iterArg_5, %iterArg, sizes = [1, 1] : (tensor<2000x2000xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
      %8 = stablehlo.reshape %7 : (tensor<1x1xf64>) -> tensor<f64>
      %9 = stablehlo.add %iterArg, %c : tensor<i64>
      %10 = stablehlo.dynamic_slice %iterArg_6, %iterArg_5, %9, sizes = [1, 1] : (tensor<2000x2000xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
      %11 = stablehlo.reshape %10 : (tensor<1x1xf64>) -> tensor<f64>
      %12 = stablehlo.add %8, %11 : tensor<f64>
      %13 = stablehlo.add %iterArg_5, %c_2 : tensor<i64>
      %14 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
      %15 = stablehlo.dynamic_slice %iterArg_6, %13, %iterArg, sizes = [1, 1] : (tensor<2000x2000xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
      %16 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
      %17 = stablehlo.select %16, %14, %15 : tensor<1x1xi1>, tensor<1x1xf64>
      %18 = stablehlo.dynamic_update_slice %iterArg_6, %17, %13, %iterArg : (tensor<2000x2000xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<2000x2000xf64>
      stablehlo.return %13, %18 : tensor<i64>, tensor<2000x2000xf64>
    }
    %2 = stablehlo.add %iterArg, %c_2 : tensor<i64>
    stablehlo.return %2, %1#1 : tensor<i64>, tensor<2000x2000xf64>
  }
  return %0#1 : tensor<2000x2000xf64>
}

// CHECK-LABEL: func.func @recurrence(
// CHECK: stablehlo.while(%[[I:.+]] = %{{.+}}, %[[OUTER:.+]] = %arg0) : tensor<i64>, tensor<2000x2000xf64>
// CHECK: } do {
// CHECK: stablehlo.while(%[[J:.+]] = %{{.+}}, %[[BUF:.+]] = %[[OUTER]]) : tensor<i64>, tensor<2000x2000xf64>
// CHECK: } do {
// CHECK-NOT: stablehlo.dynamic_slice %[[OUTER]]
// CHECK: stablehlo.select
// CHECK-NOT: stablehlo.dynamic_slice %[[OUTER]]
// CHECK: stablehlo.dynamic_update_slice %[[BUF]]

// -----

// Every read of the buffer is of the lane this iteration writes: the loop is
// a map over the lanes, and reading the loop operand frees the batching
// rewrites to lift it.
func.func @map(%arg0: tensor<16xf64>) -> tensor<16xf64> {
  %c_0 = stablehlo.constant dense<0> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_16 = stablehlo.constant dense<16> : tensor<i64>
  %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_1 = %arg0) : tensor<i64>, tensor<16xf64>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c_16 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.dynamic_slice %iterArg_1, %iterArg, sizes = [1] : (tensor<16xf64>, tensor<i64>) -> tensor<1xf64>
    %2 = stablehlo.sine %1 : tensor<1xf64>
    %3 = stablehlo.dynamic_update_slice %iterArg_1, %2, %iterArg : (tensor<16xf64>, tensor<1xf64>, tensor<i64>) -> tensor<16xf64>
    %4 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    stablehlo.return %4, %3 : tensor<i64>, tensor<16xf64>
  }
  return %0#1 : tensor<16xf64>
}

// CHECK-LABEL: func.func @map(
// CHECK-SAME: %[[X:.+]]: tensor<16xf64>
// CHECK: stablehlo.while(%{{.+}} = %{{.+}}, %[[BUF:.+]] = %[[X]])
// CHECK: } do {
// CHECK: stablehlo.dynamic_slice %[[X]], %{{.+}}, sizes = [1]
// CHECK: stablehlo.dynamic_update_slice %[[BUF]]
