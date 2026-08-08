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

// CHECK-LABEL: func.func private @diffewith_checkpointing
// The reverse sweep visits checkpoint blocks back to front, so the block's
// first iteration is outerStart = nInner * (nOuter - 1 - iterArg).
// CHECK:        %[[outerStep:.+]] = stablehlo.subtract %c, %iterArg : tensor<i64>
// CHECK-NEXT:   %[[outerStart:.+]] = stablehlo.multiply %c_2, %[[outerStep]] : tensor<i64>
// Within the block, the recompute sweep must count forward from outerStart.
// CHECK:        } do {
// CHECK-NEXT:     %[[iv:.+]] = stablehlo.add %[[outerStart]], %iterArg_18 : tensor<i64>
// and that forward-counting index is what the iv-dependent read is given
// (the loop starts at 0 and steps by 1, so the index is the iv itself).
// CHECK:          %[[ivi32:.+]] = stablehlo.convert %[[iv]] : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:     stablehlo.dynamic_slice %{{.+}}, %[[ivi32]]
