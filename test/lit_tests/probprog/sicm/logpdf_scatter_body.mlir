// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm --outline-mcmc-regions %s | FileCheck %s

// ============================================================================
// Test: Scatter body constant re-materialization after MCMC outlining
// ============================================================================
//
// MCMCRegionOp is not IsolatedFromAbove, so MLIR's CSE/canonicalization can
// move constants out of the region. After outlining, those constants become
// function arguments. Scatter bodies that reference them would fail XLA export
// (which requires scatter body return operands to be defined in the body).
//
// The fix: during outlining, re-materialize constant-like free values inside
// nested region bodies (scatter/reduce/sort).
//
// The logpdf function has a scatter whose body returns a constant (0.5).
// After inline → SICM → outline, the outlined function's scatter body must
// contain a local constant, not reference a function argument.

// CHECK-LABEL: func.func @test_scatter_body

// The outlined logpdf function should have the constant inside the scatter body
// CHECK-LABEL: func.func private @test_scatter_body_mcmc_model0
// CHECK: "stablehlo.scatter"
// CHECK: ^bb0
// CHECK: arith.constant dense<5.000000e-01>
// CHECK-NEXT: stablehlo.return

module {
  func.func private @logpdf(%pos : tensor<1x1xf64>, %A : tensor<4xf64>)
      -> tensor<f64> {
    %x = stablehlo.reshape %pos : (tensor<1x1xf64>) -> tensor<f64>
    // A constant that will be used by the scatter body
    %cst = arith.constant dense<0.5> : tensor<f64>
    // Scatter: overwrite A[0] with the constant
    %idx = arith.constant dense<[[0]]> : tensor<1x1xi64>
    %update = stablehlo.reshape %x : (tensor<f64>) -> tensor<1xf64>
    %scattered = "stablehlo.scatter"(%A, %idx, %update)
        <{indices_are_sorted = false,
          scatter_dimension_numbers = #stablehlo.scatter<
            inserted_window_dims = [0],
            scatter_dims_to_operand_dims = [0],
            index_vector_dim = 1>,
          unique_indices = true}> ({
      ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
        "stablehlo.return"(%cst) : (tensor<f64>) -> ()
    }) : (tensor<4xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<4xf64>
    // Sum and return scalar
    %zero = arith.constant dense<0.0> : tensor<f64>
    %sum = stablehlo.reduce(%scattered init: %zero) applies stablehlo.add across dimensions = [0]
        : (tensor<4xf64>, tensor<f64>) -> tensor<f64>
    return %sum : tensor<f64>
  }

  func.func @test_scatter_body(%rng : tensor<2xui64>, %A : tensor<4xf64>,
                                %pos0 : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:3 = enzyme.mcmc (%rng, %A)
        step_size = %step_size logpdf_fn = @logpdf initial_position = %pos0 {
      selection = [[]],
      all_addresses = [[]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<4xf64>, tensor<f64>, tensor<1x1xf64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
