// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | stablehlo-translate --interpret

// Test compares sqrt checkpointing vs periodic checkpointing vs no checkpointing
// N=16 iterations, period=4, sqrt(16)=4
// Computes 16! = 20,922,789,888,000 (manageable size for floating-point)

module {
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

  func.func @with_periodic_checkpointing(%arg0: tensor<f64>) -> tensor<f64> {
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

  func.func @main() {
    %input = stablehlo.constant dense<1.0000001> : tensor<f64>
    %diffe = stablehlo.constant dense<1.0> : tensor<f64>

    %diffe_sqrt:2 = enzyme.autodiff @with_sqrt_checkpointing(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    %diffe_periodic:2 = enzyme.autodiff @with_periodic_checkpointing(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    %diffe_no_checkpointing:2 = enzyme.autodiff @without_checkpointing(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    check.expect_almost_eq %diffe_sqrt#0, %diffe_no_checkpointing#0 : tensor<f64>
    check.expect_almost_eq %diffe_sqrt#1, %diffe_no_checkpointing#1 : tensor<f64>
    check.expect_almost_eq %diffe_periodic#0, %diffe_no_checkpointing#0 : tensor<f64>
    check.expect_almost_eq %diffe_periodic#1, %diffe_no_checkpointing#1 : tensor<f64>
    check.expect_almost_eq %diffe_sqrt#0, %diffe_periodic#0 : tensor<f64>
    check.expect_almost_eq %diffe_sqrt#1, %diffe_periodic#1 : tensor<f64>

    return
  }
}
