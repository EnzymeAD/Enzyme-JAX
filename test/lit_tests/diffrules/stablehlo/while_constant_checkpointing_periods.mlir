// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | stablehlo-translate --interpret

// Test for CONSTANT_CHECKPOINTING with different checkpoint periods
// This test verifies that checkpointing works correctly with:
// 1. Default sqrt checkpointing (enable_checkpointing = true, no period)
// 2. Explicit periods: 2, 3, 5, 7
// 3. Edge cases: period = 1, period = numIters
// 4. Both evenly divisible and non-evenly divisible cases

module {
  // Reference function without any checkpointing
  func.func @without_checkpointing(%arg0: tensor<f64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<16> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
      %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
      stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }

  // Default sqrt checkpointing (16 iterations, sqrt(16) = 4)
  func.func @with_sqrt_checkpointing(%arg0: tensor<f64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<16> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut, enzymexla.enable_checkpointing = true}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
      %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
      stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }

  // Period = 2 (16 iterations / 2 = 8 outer iterations, evenly divisible)
  func.func @with_period_2(%arg0: tensor<f64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<16> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut, enzymexla.checkpoint_period = 2 : i64}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
      %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
      stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }

  // Period = 3 (16 iterations / 3 = 6 outer iterations (5*3 + 1), not evenly divisible)
  func.func @with_period_3(%arg0: tensor<f64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<16> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut, enzymexla.checkpoint_period = 3 : i64}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
      %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
      stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }

  // Period = 5 (16 iterations / 5 = 4 outer iterations (3*5 + 1), not evenly divisible)
  func.func @with_period_5(%arg0: tensor<f64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<16> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut, enzymexla.checkpoint_period = 5 : i64}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
      %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
      stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }

  // Period = 7 (16 iterations / 7 = 3 outer iterations (2*7 + 2), not evenly divisible)
  func.func @with_period_7(%arg0: tensor<f64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<16> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut, enzymexla.checkpoint_period = 7 : i64}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
      %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
      stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }

  // Period = 1 (edge case: checkpoint every iteration, 16 outer iterations)
  func.func @with_period_1(%arg0: tensor<f64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<16> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut, enzymexla.checkpoint_period = 1 : i64}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
      %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
      stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }

  // Period = 4 (16 iterations / 4 = 4 outer iterations, evenly divisible, matches sqrt(16) = 4)
  func.func @with_period_4(%arg0: tensor<f64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<16> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut, enzymexla.checkpoint_period = 4 : i64}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
      %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
      stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }

  // Period = 8 (16 iterations / 8 = 2 outer iterations, evenly divisible)
  func.func @with_period_8(%arg0: tensor<f64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<16> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut, enzymexla.checkpoint_period = 8 : i64}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
      %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
      stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }

  // Period = 16 (edge case: period equals numIters, 1 outer iteration)
  func.func @with_period_16(%arg0: tensor<f64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<16> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut, enzymexla.checkpoint_period = 16 : i64}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
      %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
      stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }

  func.func @main() {
    %input = stablehlo.constant dense<1.0000001> : tensor<f64>
    %diffe = stablehlo.constant dense<1.0> : tensor<f64>

    // Compute reference gradient without checkpointing
    %diffe_no_checkpointing:2 = enzyme.autodiff @without_checkpointing(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    // Test default sqrt checkpointing
    %diffe_sqrt:2 = enzyme.autodiff @with_sqrt_checkpointing(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    // Test different periods
    %diffe_period_2:2 = enzyme.autodiff @with_period_2(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    %diffe_period_3:2 = enzyme.autodiff @with_period_3(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    %diffe_period_5:2 = enzyme.autodiff @with_period_5(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    %diffe_period_7:2 = enzyme.autodiff @with_period_7(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    %diffe_period_1:2 = enzyme.autodiff @with_period_1(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    %diffe_period_4:2 = enzyme.autodiff @with_period_4(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    %diffe_period_8:2 = enzyme.autodiff @with_period_8(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    %diffe_period_16:2 = enzyme.autodiff @with_period_16(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    // Verify all checkpointing strategies produce the same results as no checkpointing
    check.expect_almost_eq %diffe_sqrt#0, %diffe_no_checkpointing#0 : tensor<f64>
    check.expect_almost_eq %diffe_sqrt#1, %diffe_no_checkpointing#1 : tensor<f64>

    check.expect_almost_eq %diffe_period_2#0, %diffe_no_checkpointing#0 : tensor<f64>
    check.expect_almost_eq %diffe_period_2#1, %diffe_no_checkpointing#1 : tensor<f64>

    check.expect_almost_eq %diffe_period_3#0, %diffe_no_checkpointing#0 : tensor<f64>
    check.expect_almost_eq %diffe_period_3#1, %diffe_no_checkpointing#1 : tensor<f64>

    check.expect_almost_eq %diffe_period_5#0, %diffe_no_checkpointing#0 : tensor<f64>
    check.expect_almost_eq %diffe_period_5#1, %diffe_no_checkpointing#1 : tensor<f64>

    check.expect_almost_eq %diffe_period_7#0, %diffe_no_checkpointing#0 : tensor<f64>
    check.expect_almost_eq %diffe_period_7#1, %diffe_no_checkpointing#1 : tensor<f64>

    check.expect_almost_eq %diffe_period_1#0, %diffe_no_checkpointing#0 : tensor<f64>
    check.expect_almost_eq %diffe_period_1#1, %diffe_no_checkpointing#1 : tensor<f64>

    check.expect_almost_eq %diffe_period_4#0, %diffe_no_checkpointing#0 : tensor<f64>
    check.expect_almost_eq %diffe_period_4#1, %diffe_no_checkpointing#1 : tensor<f64>

    check.expect_almost_eq %diffe_period_8#0, %diffe_no_checkpointing#0 : tensor<f64>
    check.expect_almost_eq %diffe_period_8#1, %diffe_no_checkpointing#1 : tensor<f64>

    check.expect_almost_eq %diffe_period_16#0, %diffe_no_checkpointing#0 : tensor<f64>
    check.expect_almost_eq %diffe_period_16#1, %diffe_no_checkpointing#1 : tensor<f64>

    // Verify that sqrt checkpointing matches period=4 (since sqrt(16) = 4)
    check.expect_almost_eq %diffe_sqrt#0, %diffe_period_4#0 : tensor<f64>
    check.expect_almost_eq %diffe_sqrt#1, %diffe_period_4#1 : tensor<f64>

    return
  }
}
