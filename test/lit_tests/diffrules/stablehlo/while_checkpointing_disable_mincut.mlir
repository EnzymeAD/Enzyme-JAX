// RUN: enzymexlamlir-opt %s --enzyme --canonicalize | FileCheck %s

// Checkpointing replaces one loop with several, and each of those is still
// doing the original loop's work. A directive such as enzyme.disable_mincut
// therefore has to come along -- otherwise asking for the min cut to be off
// applies only to the loop the user wrote and is silently dropped for every
// loop derived from it.

module {
  func.func @ckpt(%arg0: tensor<f64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<12> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut, enzyme.checkpoint_period = 4 : i64, enzyme.enable_checkpointing = true}
     cond {
      %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
      %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
      stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }

  func.func @ckpt_diff(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
    %d:2 = enzyme.autodiff @ckpt(%arg0, %arg1) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    return %d#0, %d#1 : tensor<f64>, tensor<f64>
  }
}

// Checkpointing splits the loop into several, and every one of them keeps the
// directive. The checkpointing attributes are not copied along, since the split
// has already happened.
// CHECK-LABEL: func.func @ckpt_diff
// CHECK-COUNT-4: enzyme.disable_mincut
// CHECK-NOT: enzyme.enable_checkpointing
