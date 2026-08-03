// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | stablehlo-translate --interpret

// Regression test: the body reads a constant tensor indexed by the induction
// variable, and accumulates a per-iteration contribution. Unlike a plain
// product loop, this is sensitive to the order in which the iterations of a
// checkpoint block are replayed, so it catches a recompute sweep that walks
// its induction variable backwards.
//
// The recompute sweep inside a checkpoint block replays the primal forward, so
// its induction variable must be outerStart + idx. Walking it backwards
// (outerStart + nInner - 1 - idx) leaves the loop-carried state correct but
// feeds every iv-dependent read of the block the wrong index.

module {
  func.func private @without_checkpointing(%a: tensor<f64>, %c: tensor<12xf64>) -> tensor<f64> {
    %step = stablehlo.constant dense<1> : tensor<i64>
    %n = stablehlo.constant dense<12> : tensor<i64>
    %zeroi = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0:5 = stablehlo.while(%iterArg = %zeroi, %arg_a = %a, %arg_c = %c, %arg_y = %one, %arg_acc = %zero) : tensor<i64>, tensor<f64>, tensor<12xf64>, tensor<f64>, tensor<f64> attributes {enzyme.disable_mincut}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %step : tensor<i64>
      %2 = stablehlo.multiply %arg_a, %arg_y : tensor<f64>
      %3 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.dynamic_slice %arg_c, %3, sizes = [1] : (tensor<12xf64>, tensor<i32>) -> tensor<1xf64>
      %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
      %6 = stablehlo.multiply %5, %2 : tensor<f64>
      %7 = stablehlo.add %arg_acc, %6 : tensor<f64>
      stablehlo.return %1, %arg_a, %arg_c, %2, %7 : tensor<i64>, tensor<f64>, tensor<12xf64>, tensor<f64>, tensor<f64>
    }
    return %0#4 : tensor<f64>
  }

  func.func private @with_checkpointing(%a: tensor<f64>, %c: tensor<12xf64>) -> tensor<f64> {
    %step = stablehlo.constant dense<1> : tensor<i64>
    %n = stablehlo.constant dense<12> : tensor<i64>
    %zeroi = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0:5 = stablehlo.while(%iterArg = %zeroi, %arg_a = %a, %arg_c = %c, %arg_y = %one, %arg_acc = %zero) : tensor<i64>, tensor<f64>, tensor<12xf64>, tensor<f64>, tensor<f64> attributes {enzyme.disable_mincut, enzymexla.enable_checkpointing = true, enzymexla.checkpoint_period = 4 : i64}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %step : tensor<i64>
      %2 = stablehlo.multiply %arg_a, %arg_y : tensor<f64>
      %3 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.dynamic_slice %arg_c, %3, sizes = [1] : (tensor<12xf64>, tensor<i32>) -> tensor<1xf64>
      %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
      %6 = stablehlo.multiply %5, %2 : tensor<f64>
      %7 = stablehlo.add %arg_acc, %6 : tensor<f64>
      stablehlo.return %1, %arg_a, %arg_c, %2, %7 : tensor<i64>, tensor<f64>, tensor<12xf64>, tensor<f64>, tensor<f64>
    }
    return %0#4 : tensor<f64>
  }

  func.func @main() {
    %a = stablehlo.constant dense<1.100000e+00> : tensor<f64>
    %c = stablehlo.constant dense<[1.000000e+00, -2.000000e+00, 3.000000e+00, 5.000000e-01, -1.500000e+00, 7.000000e-01, 2.500000e+00, -3.000000e-01, 1.200000e+00, -8.000000e-01, 4.000000e-01, 2.000000e+00]> : tensor<12xf64>
    %diffe = stablehlo.constant dense<1.000000e+00> : tensor<f64>

    %ckpt:2 = enzyme.autodiff @with_checkpointing(%a, %c, %diffe) {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<12xf64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    %plain:2 = enzyme.autodiff @without_checkpointing(%a, %c, %diffe) {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<12xf64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    check.expect_almost_eq %ckpt#0, %plain#0 : tensor<f64>
    check.expect_almost_eq %ckpt#1, %plain#1 : tensor<f64>

    return
  }
}

// The reverse sweep visits checkpoint blocks back to front, so the block's
// first iteration is outerStart = nInner * (nOuter - 1 - iterArg).
// Within the block, the recompute sweep must count forward from outerStart.

// CHECK: module {
// CHECK-NEXT:   func.func private @without_checkpointing(%arg0: tensor<f64>, %arg1: tensor<12xf64>) -> tensor<f64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<12> : tensor<i64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:     %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0:5 = stablehlo.while(%iterArg = %c_1, %iterArg_3 = %arg0, %iterArg_4 = %arg1, %iterArg_5 = %cst, %iterArg_6 = %cst_2) : tensor<i64>, tensor<f64>, tensor<12xf64>, tensor<f64>, tensor<f64> attributes {enzyme.disable_mincut}
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %1 = stablehlo.add %iterArg, %c : tensor<i64>
// CHECK-NEXT:       %2 = stablehlo.multiply %iterArg_3, %iterArg_5 : tensor<f64>
// CHECK-NEXT:       %3 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:       %4 = stablehlo.dynamic_slice %iterArg_4, %3, sizes = [1] : (tensor<12xf64>, tensor<i32>) -> tensor<1xf64>
// CHECK-NEXT:       %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:       %6 = stablehlo.multiply %5, %2 : tensor<f64>
// CHECK-NEXT:       %7 = stablehlo.add %iterArg_6, %6 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %1, %iterArg_3, %iterArg_4, %2, %7 : tensor<i64>, tensor<f64>, tensor<12xf64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#4 : tensor<f64>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @with_checkpointing(%arg0: tensor<f64>, %arg1: tensor<12xf64>) -> tensor<f64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<12> : tensor<i64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:     %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0:5 = stablehlo.while(%iterArg = %c_1, %iterArg_3 = %arg0, %iterArg_4 = %arg1, %iterArg_5 = %cst, %iterArg_6 = %cst_2) : tensor<i64>, tensor<f64>, tensor<12xf64>, tensor<f64>, tensor<f64> attributes {enzyme.disable_mincut, enzymexla.checkpoint_period = 4 : i64, enzymexla.enable_checkpointing = true}
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %1 = stablehlo.add %iterArg, %c : tensor<i64>
// CHECK-NEXT:       %2 = stablehlo.multiply %iterArg_3, %iterArg_5 : tensor<f64>
// CHECK-NEXT:       %3 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:       %4 = stablehlo.dynamic_slice %iterArg_4, %3, sizes = [1] : (tensor<12xf64>, tensor<i32>) -> tensor<1xf64>
// CHECK-NEXT:       %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:       %6 = stablehlo.multiply %5, %2 : tensor<f64>
// CHECK-NEXT:       %7 = stablehlo.add %iterArg_6, %6 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %1, %iterArg_3, %iterArg_4, %2, %7 : tensor<i64>, tensor<f64>, tensor<12xf64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#4 : tensor<f64>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @main() {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.100000e+00> : tensor<f64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<[1.000000e+00, -2.000000e+00, 3.000000e+00, 5.000000e-01, -1.500000e+00, 0.69999999999999996, 2.500000e+00, -3.000000e-01, 1.200000e+00, -8.000000e-01, 4.000000e-01, 2.000000e+00]> : tensor<12xf64>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0:2 = call @diffewith_checkpointing(%cst, %cst_0, %cst_1) : (tensor<f64>, tensor<12xf64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
// CHECK-NEXT:     %1:2 = call @diffewithout_checkpointing(%cst, %cst_0, %cst_1) : (tensor<f64>, tensor<12xf64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
// CHECK-NEXT:     check.expect_almost_eq %0#0, %1#0 : tensor<f64>
// CHECK-NEXT:     check.expect_almost_eq %0#1, %1#1 : tensor<f64>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @diffewith_checkpointing(%arg0: tensor<f64>, %arg1: tensor<12xf64>, %arg2: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<3x12xf64>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<3xf64>
// CHECK-NEXT:     %c = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:     %c_3 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:     %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:     %c_6 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %c_7 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %0:9 = stablehlo.while(%iterArg = %c_6, %iterArg_8 = %arg0, %iterArg_9 = %arg1, %iterArg_10 = %cst_5, %iterArg_11 = %cst_4, %iterArg_12 = %cst_1, %iterArg_13 = %cst_0, %iterArg_14 = %cst_1, %iterArg_15 = %cst_1) : tensor<i64>, tensor<f64>, tensor<12xf64>, tensor<f64>, tensor<f64>, tensor<3xf64>, tensor<3x12xf64>, tensor<3xf64>, tensor<3xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %2 = stablehlo.compare LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %2 = stablehlo.reshape %iterArg_8 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:       %3 = stablehlo.dynamic_update_slice %iterArg_15, %2, %iterArg : (tensor<3xf64>, tensor<1xf64>, tensor<i64>) -> tensor<3xf64>
// CHECK-NEXT:       %4 = stablehlo.reshape %iterArg_10 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:       %5 = stablehlo.dynamic_update_slice %iterArg_14, %4, %iterArg : (tensor<3xf64>, tensor<1xf64>, tensor<i64>) -> tensor<3xf64>
// CHECK-NEXT:       %6 = stablehlo.reshape %iterArg_9 : (tensor<12xf64>) -> tensor<1x12xf64>
// CHECK-NEXT:       %7 = stablehlo.dynamic_update_slice %iterArg_13, %6, %iterArg, %c_6 : (tensor<3x12xf64>, tensor<1x12xf64>, tensor<i64>, tensor<i64>) -> tensor<3x12xf64>
// CHECK-NEXT:       %8 = stablehlo.reshape %iterArg_11 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:       %9 = stablehlo.dynamic_update_slice %iterArg_12, %8, %iterArg : (tensor<3xf64>, tensor<1xf64>, tensor<i64>) -> tensor<3xf64>
// CHECK-NEXT:       %10 = stablehlo.multiply %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:       %11:5 = stablehlo.while(%iterArg_16 = %c_6, %iterArg_17 = %iterArg_8, %iterArg_18 = %iterArg_9, %iterArg_19 = %iterArg_10, %iterArg_20 = %iterArg_11) : tensor<i64>, tensor<f64>, tensor<12xf64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:         %13 = stablehlo.compare LT, %iterArg_16, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:         stablehlo.return %13 : tensor<i1>
// CHECK-NEXT:       } do {
// CHECK-NEXT:         %13 = stablehlo.add %iterArg_16, %10 : tensor<i64>
// CHECK-NEXT:         %14 = stablehlo.multiply %iterArg_17, %iterArg_19 : tensor<f64>
// CHECK-NEXT:         %15 = stablehlo.convert %13 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:         %16 = stablehlo.dynamic_slice %iterArg_18, %15, sizes = [1] : (tensor<12xf64>, tensor<i32>) -> tensor<1xf64>
// CHECK-NEXT:         %17 = stablehlo.reshape %16 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:         %18 = stablehlo.multiply %17, %14 : tensor<f64>
// CHECK-NEXT:         %19 = stablehlo.add %iterArg_20, %18 : tensor<f64>
// CHECK-NEXT:         %20 = stablehlo.add %iterArg_16, %c_7 : tensor<i64>
// CHECK-NEXT:         stablehlo.return %20, %iterArg_17, %iterArg_18, %14, %19 : tensor<i64>, tensor<f64>, tensor<12xf64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:       }
// CHECK-NEXT:       %12 = stablehlo.add %iterArg, %c_7 : tensor<i64>
// CHECK-NEXT:       stablehlo.return %12, %11#1, %11#2, %11#3, %11#4, %9, %7, %5, %3 : tensor<i64>, tensor<f64>, tensor<12xf64>, tensor<f64>, tensor<f64>, tensor<3xf64>, tensor<3x12xf64>, tensor<3xf64>, tensor<3xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %1:10 = stablehlo.while(%iterArg = %c_6, %iterArg_8 = %cst_4, %iterArg_9 = %cst_4, %iterArg_10 = %arg2, %iterArg_11 = %cst_4, %iterArg_12 = %cst_4, %iterArg_13 = %cst_4, %iterArg_14 = %cst_4, %iterArg_15 = %cst_4, %iterArg_16 = %cst_4) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %2 = stablehlo.compare LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %2 = stablehlo.subtract %c, %iterArg : tensor<i64>
// CHECK-NEXT:       %3 = stablehlo.dynamic_slice %0#5, %2, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:       %4 = stablehlo.reshape %3 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:       %5 = stablehlo.dynamic_slice %0#6, %2, %c_6, sizes = [1, 12] : (tensor<3x12xf64>, tensor<i64>, tensor<i64>) -> tensor<1x12xf64>
// CHECK-NEXT:       %6 = stablehlo.reshape %5 : (tensor<1x12xf64>) -> tensor<12xf64>
// CHECK-NEXT:       %7 = stablehlo.dynamic_slice %0#7, %2, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:       %8 = stablehlo.reshape %7 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:       %9 = stablehlo.dynamic_slice %0#8, %2, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:       %10 = stablehlo.reshape %9 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:       %11 = stablehlo.subtract %c, %iterArg : tensor<i64>
// CHECK-NEXT:       %12 = stablehlo.multiply %c_2, %11 : tensor<i64>
// CHECK-NEXT:       %13:8 = stablehlo.while(%iterArg_17 = %c_6, %iterArg_18 = %10, %iterArg_19 = %6, %iterArg_20 = %8, %iterArg_21 = %4, %iterArg_22 = %cst, %iterArg_23 = %cst, %iterArg_24 = %cst) : tensor<i64>, tensor<f64>, tensor<12xf64>, tensor<f64>, tensor<f64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:         %16 = stablehlo.compare LT, %iterArg_17, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:         stablehlo.return %16 : tensor<i1>
// CHECK-NEXT:       } do {
// CHECK-NEXT:         %16 = stablehlo.add %12, %iterArg_17 : tensor<i64>
// CHECK-NEXT:         %17 = stablehlo.multiply %c_7, %16 : tensor<i64>
// CHECK-NEXT:         %18 = stablehlo.reshape %iterArg_18 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:         %19 = stablehlo.dynamic_update_slice %iterArg_22, %18, %iterArg_17 : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
// CHECK-NEXT:         %20 = stablehlo.reshape %iterArg_20 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:         %21 = stablehlo.dynamic_update_slice %iterArg_23, %20, %iterArg_17 : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
// CHECK-NEXT:         %22 = stablehlo.multiply %iterArg_18, %iterArg_20 : tensor<f64>
// CHECK-NEXT:         %23 = stablehlo.convert %17 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:         %24 = stablehlo.dynamic_slice %iterArg_19, %23, sizes = [1] : (tensor<12xf64>, tensor<i32>) -> tensor<1xf64>
// CHECK-NEXT:         %25 = stablehlo.reshape %24 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:         %26 = stablehlo.reshape %25 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:         %27 = stablehlo.dynamic_update_slice %iterArg_24, %26, %iterArg_17 : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
// CHECK-NEXT:         %28 = stablehlo.multiply %25, %22 : tensor<f64>
// CHECK-NEXT:         %29 = stablehlo.add %iterArg_21, %28 : tensor<f64>
// CHECK-NEXT:         %30 = stablehlo.add %iterArg_17, %c_7 : tensor<i64>
// CHECK-NEXT:         stablehlo.return %30, %iterArg_18, %iterArg_19, %22, %29, %19, %21, %27 : tensor<i64>, tensor<f64>, tensor<12xf64>, tensor<f64>, tensor<f64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:       %14:10 = stablehlo.while(%iterArg_17 = %c_6, %iterArg_18 = %iterArg_8, %iterArg_19 = %iterArg_9, %iterArg_20 = %iterArg_10, %iterArg_21 = %iterArg_11, %iterArg_22 = %iterArg_12, %iterArg_23 = %iterArg_13, %iterArg_24 = %iterArg_14, %iterArg_25 = %iterArg_15, %iterArg_26 = %iterArg_16) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:         %16 = stablehlo.compare LT, %iterArg_17, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:         stablehlo.return %16 : tensor<i1>
// CHECK-NEXT:       } do {
// CHECK-NEXT:         %16 = stablehlo.subtract %c_3, %iterArg_17 : tensor<i64>
// CHECK-NEXT:         %17 = stablehlo.add %iterArg_21, %iterArg_18 : tensor<f64>
// CHECK-NEXT:         %18 = stablehlo.add %iterArg_22, %iterArg_19 : tensor<f64>
// CHECK-NEXT:         %19 = stablehlo.add %iterArg_23, %iterArg_20 : tensor<f64>
// CHECK-NEXT:         %20 = stablehlo.add %iterArg_24, %19 : tensor<f64>
// CHECK-NEXT:         %21 = stablehlo.add %iterArg_25, %19 : tensor<f64>
// CHECK-NEXT:         %22 = stablehlo.dynamic_slice %13#7, %16, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:         %23 = stablehlo.reshape %22 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:         %24 = stablehlo.multiply %21, %23 : tensor<f64>
// CHECK-NEXT:         %25 = stablehlo.add %18, %24 : tensor<f64>
// CHECK-NEXT:         %26 = stablehlo.dynamic_slice %13#5, %16, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:         %27 = stablehlo.reshape %26 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:         %28 = stablehlo.dynamic_slice %13#6, %16, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:         %29 = stablehlo.reshape %28 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:         %30 = stablehlo.multiply %25, %29 : tensor<f64>
// CHECK-NEXT:         %31 = stablehlo.add %17, %30 : tensor<f64>
// CHECK-NEXT:         %32 = stablehlo.multiply %25, %27 : tensor<f64>
// CHECK-NEXT:         %33 = stablehlo.add %iterArg_26, %32 : tensor<f64>
// CHECK-NEXT:         %34 = stablehlo.add %iterArg_17, %c_7 : tensor<i64>
// CHECK-NEXT:         stablehlo.return %34, %31, %33, %20, %cst_4, %cst_4, %cst_4, %cst_4, %cst_4, %cst_4 : tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:       }
// CHECK-NEXT:       %15 = stablehlo.add %iterArg, %c_7 : tensor<i64>
// CHECK-NEXT:       stablehlo.return %15, %14#1, %14#2, %14#3, %14#4, %14#5, %14#6, %14#7, %14#8, %14#9 : tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#4, %1#1 : tensor<f64>, tensor<f64>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @diffewithout_checkpointing(%arg0: tensor<f64>, %arg1: tensor<12xf64>, %arg2: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
// CHECK-NEXT:     %c = stablehlo.constant dense<11> : tensor<i64>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<12xf64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %c_3 = stablehlo.constant dense<12> : tensor<i64>
// CHECK-NEXT:     %c_4 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %0:8 = stablehlo.while(%iterArg = %c_2, %iterArg_5 = %arg0, %iterArg_6 = %arg1, %iterArg_7 = %cst_1, %iterArg_8 = %cst_0, %iterArg_9 = %cst, %iterArg_10 = %cst, %iterArg_11 = %cst) : tensor<i64>, tensor<f64>, tensor<12xf64>, tensor<f64>, tensor<f64>, tensor<12xf64>, tensor<12xf64>, tensor<12xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %2 = stablehlo.compare LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %2 = stablehlo.add %iterArg, %c_4 : tensor<i64>
// CHECK-NEXT:       %3 = stablehlo.reshape %iterArg_5 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:       %4 = stablehlo.dynamic_update_slice %iterArg_9, %3, %iterArg : (tensor<12xf64>, tensor<1xf64>, tensor<i64>) -> tensor<12xf64>
// CHECK-NEXT:       %5 = stablehlo.reshape %iterArg_7 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:       %6 = stablehlo.dynamic_update_slice %iterArg_10, %5, %iterArg : (tensor<12xf64>, tensor<1xf64>, tensor<i64>) -> tensor<12xf64>
// CHECK-NEXT:       %7 = stablehlo.multiply %iterArg_5, %iterArg_7 : tensor<f64>
// CHECK-NEXT:       %8 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:       %9 = stablehlo.dynamic_slice %iterArg_6, %8, sizes = [1] : (tensor<12xf64>, tensor<i32>) -> tensor<1xf64>
// CHECK-NEXT:       %10 = stablehlo.reshape %9 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:       %11 = stablehlo.reshape %10 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:       %12 = stablehlo.dynamic_update_slice %iterArg_11, %11, %iterArg : (tensor<12xf64>, tensor<1xf64>, tensor<i64>) -> tensor<12xf64>
// CHECK-NEXT:       %13 = stablehlo.multiply %10, %7 : tensor<f64>
// CHECK-NEXT:       %14 = stablehlo.add %iterArg_8, %13 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %2, %iterArg_5, %iterArg_6, %7, %14, %4, %6, %12 : tensor<i64>, tensor<f64>, tensor<12xf64>, tensor<f64>, tensor<f64>, tensor<12xf64>, tensor<12xf64>, tensor<12xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %1:4 = stablehlo.while(%iterArg = %c_2, %iterArg_5 = %cst_0, %iterArg_6 = %cst_0, %iterArg_7 = %arg2) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %2 = stablehlo.compare LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %2 = stablehlo.subtract %c, %iterArg : tensor<i64>
// CHECK-NEXT:       %3 = stablehlo.add %iterArg, %c_4 : tensor<i64>
// CHECK-NEXT:       %4 = stablehlo.dynamic_slice %0#7, %2, sizes = [1] : (tensor<12xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:       %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:       %6 = stablehlo.multiply %iterArg_7, %5 : tensor<f64>
// CHECK-NEXT:       %7 = stablehlo.add %iterArg_6, %6 : tensor<f64>
// CHECK-NEXT:       %8 = stablehlo.dynamic_slice %0#5, %2, sizes = [1] : (tensor<12xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:       %9 = stablehlo.reshape %8 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:       %10 = stablehlo.dynamic_slice %0#6, %2, sizes = [1] : (tensor<12xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:       %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:       %12 = stablehlo.multiply %7, %11 : tensor<f64>
// CHECK-NEXT:       %13 = stablehlo.add %iterArg_5, %12 : tensor<f64>
// CHECK-NEXT:       %14 = stablehlo.multiply %7, %9 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %3, %13, %14, %iterArg_7 : tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#4, %1#1 : tensor<f64>, tensor<f64>
// CHECK-NEXT:   }
// CHECK-NEXT: }
