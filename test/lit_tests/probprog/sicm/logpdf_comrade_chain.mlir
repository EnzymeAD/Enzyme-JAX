// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: Comrade NUFFT chain in logpdf mode (custom logpdf_fn)
// ============================================================================
//
// Same NUFFT chain as comrade_chain.mlir but using logpdf_fn + initial_position
// instead of fn + enzyme.sample. This is the code path used by mcmc_logpdf().
//
// The logpdf function takes (position:[1,1], A, weights, idx) → f64 and
// computes the same chain:
//   broadcast(x) → convert → multiply(·, weights) →
//   scatter(zeros, idx, ·) → complex → transpose → FFT → reshape →
//   dot_general(A, ·) → scalar
//
// Here: M=3, K=2, N1=N2=2, P=4
//
// Expected SICM fixpoint (same as trace-based test):
//   Iter 1: DotAbsorbFFT   → A' = fft(reshape(A))
//   Iter 2: DotAbsorbTranspose → A'' = transpose(A')
//   Iter 3: DotAbsorbScatter   → A''' = gather(A'', idx)

// CHECK-LABEL: func.func @test_logpdf_comrade
// CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>,
// CHECK-SAME: %[[A:.*]]: tensor<3x4xcomplex<f64>>,
// CHECK-SAME: %[[WEIGHTS:.*]]: tensor<2xcomplex<f64>>,
// CHECK-SAME: %[[IDX:.*]]: tensor<2x2xi32>,
// CHECK-SAME: %[[POS0:.*]]: tensor<1x1xf64>

// Invariant ops should be hoisted: reshape, fft, reshape, reshape, transpose, reshape, gather
// CHECK: stablehlo.reshape
// CHECK: stablehlo.fft
// CHECK: stablehlo.reshape
// CHECK: stablehlo.reshape
// CHECK: stablehlo.transpose
// CHECK: stablehlo.reshape
// CHECK: stablehlo.gather
// CHECK-NOT: stablehlo.fft
// CHECK-NOT: stablehlo.transpose

// No FFT, transpose, or scatter inside mcmc_region
// CHECK: enzyme.mcmc_region
// In logpdf mode there are no sample_region ops
// CHECK-NOT: enzyme.sample_region
// CHECK-NOT: stablehlo.fft
// CHECK-NOT: stablehlo.transpose
// CHECK-NOT: stablehlo.scatter
// The dot_general should use the gathered (smaller) matrix
// CHECK: stablehlo.dot_general
// CHECK: enzyme.yield

module {
  // Logpdf function: same chain as model_comrade but position is an argument
  func.func private @comrade_logpdf(%pos : tensor<1x1xf64>,
                                     %A : tensor<3x4xcomplex<f64>>,
                                     %weights : tensor<2xcomplex<f64>>,
                                     %idx : tensor<2x2xi32>) -> tensor<f64> {
    // Extract scalar from position
    %x = stablehlo.reshape %pos : (tensor<1x1xf64>) -> tensor<f64>

    // Broadcast sample to pixel vector
    %x_vec = stablehlo.broadcast_in_dim %x, dims = [] : (tensor<f64>) -> tensor<2xf64>

    // Convert real → complex
    %x_complex = stablehlo.convert %x_vec : (tensor<2xf64>) -> tensor<2xcomplex<f64>>

    // Multiply by invariant complex weights
    %weighted = stablehlo.multiply %x_complex, %weights : tensor<2xcomplex<f64>>

    // Extract real and imaginary parts
    %re = stablehlo.real %weighted : (tensor<2xcomplex<f64>>) -> tensor<2xf64>
    %im = stablehlo.imag %weighted : (tensor<2xcomplex<f64>>) -> tensor<2xf64>

    // Scatter into 2×2 grids using invariant indices
    %zeros = stablehlo.constant dense<0.0> : tensor<2x2xf64>
    %scatter_re = "stablehlo.scatter"(%zeros, %idx, %re) <{
      scatter_dimension_numbers = #stablehlo.scatter<
        inserted_window_dims = [0, 1],
        scatter_dims_to_operand_dims = [0, 1],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      unique_indices = true
    }> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg1 : tensor<f64>
    }) : (tensor<2x2xf64>, tensor<2x2xi32>, tensor<2xf64>) -> tensor<2x2xf64>

    %scatter_im = "stablehlo.scatter"(%zeros, %idx, %im) <{
      scatter_dimension_numbers = #stablehlo.scatter<
        inserted_window_dims = [0, 1],
        scatter_dims_to_operand_dims = [0, 1],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      unique_indices = true
    }> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg1 : tensor<f64>
    }) : (tensor<2x2xf64>, tensor<2x2xi32>, tensor<2xf64>) -> tensor<2x2xf64>

    // Combine into complex grid
    %grid = stablehlo.complex %scatter_re, %scatter_im : tensor<2x2xcomplex<f64>>

    // Transpose
    %transposed = stablehlo.transpose %grid, dims = [1, 0] : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>

    // FFT
    %fft_out = stablehlo.fft %transposed, type = FFT, length = [2, 2] : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>

    // Reshape to [4, 1]
    %reshaped = stablehlo.reshape %fft_out : (tensor<2x2xcomplex<f64>>) -> tensor<4x1xcomplex<f64>>

    // dot_general(A, reshaped) — the expensive NUFFT matrix multiply
    %vis = stablehlo.dot_general %A, %reshaped, contracting_dims = [1] x [0]
        : (tensor<3x4xcomplex<f64>>, tensor<4x1xcomplex<f64>>) -> tensor<3x1xcomplex<f64>>

    // Extract a scalar result
    %real_vis = stablehlo.real %vis : (tensor<3x1xcomplex<f64>>) -> tensor<3x1xf64>
    %slice = "stablehlo.slice"(%real_vis) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 1, 1>, strides = array<i64: 1, 1>} : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %result = stablehlo.reshape %slice : (tensor<1x1xf64>) -> tensor<f64>

    return %result : tensor<f64>
  }

  func.func @test_logpdf_comrade(%rng : tensor<2xui64>,
                                  %A : tensor<3x4xcomplex<f64>>,
                                  %weights : tensor<2xcomplex<f64>>,
                                  %idx : tensor<2x2xi32>,
                                  %pos0 : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:3 = enzyme.mcmc (%rng, %A, %weights, %idx)
        step_size = %step_size logpdf_fn = @comrade_logpdf initial_position = %pos0 {
      selection = [[]],
      all_addresses = [[]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<3x4xcomplex<f64>>, tensor<2xcomplex<f64>>, tensor<2x2xi32>, tensor<f64>, tensor<1x1xf64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
