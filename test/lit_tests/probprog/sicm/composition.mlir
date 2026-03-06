// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: FFT + Transpose composition (2 fixpoint iterations)
// ============================================================================
//
// Model: dot_general(A:[2,4], reshape(fft(transpose(X:[2,2], [1,0]))) -> [4,1])
//        where A is sample-invariant.
//
// Iteration 1: DotAbsorbFFT fires, absorbs FFT into A
// Iteration 2: DotAbsorbTranspose fires, absorbs transpose into A'

// CHECK-LABEL: func.func @test_fft_transpose_composition
// CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[PRIOR:.*]]: tensor<f64>, %[[A:.*]]: tensor<2x4xcomplex<f64>>, %[[TRACE:.*]]: tensor<1x1xf64>
// Both FFT and transpose of A should be hoisted
// CHECK: stablehlo.reshape
// CHECK: stablehlo.fft
// CHECK: stablehlo.reshape
// CHECK: stablehlo.reshape
// CHECK: stablehlo.transpose
// CHECK: stablehlo.reshape
// No FFT or transpose inside mcmc_region
// CHECK: enzyme.mcmc_region
// CHECK: enzyme.sample_region
// CHECK-NOT: stablehlo.fft
// CHECK-NOT: stablehlo.transpose
// CHECK: stablehlo.dot_general
// CHECK: enzyme.yield

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %sample = arith.addf %mean, %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }

  func.func private @normal_logpdf(%x : tensor<f64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> tensor<f64> {
    %neg = arith.negf %x : tensor<f64>
    return %neg : tensor<f64>
  }

  func.func @test_fft_transpose_composition(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                             %A : tensor<2x4xcomplex<f64>>,
                                             %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_fft_transpose(%rng, %prior, %A) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<2x4xcomplex<f64>>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_fft_transpose(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                  %A : tensor<2x4xcomplex<f64>>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %x:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Build a 2x2 complex matrix
    %x_complex = stablehlo.complex %x#1, %x#1 : tensor<complex<f64>>
    %x_mat = stablehlo.broadcast_in_dim %x_complex, dims = [] : (tensor<complex<f64>>) -> tensor<2x2xcomplex<f64>>

    // Transpose
    %transposed = stablehlo.transpose %x_mat, dims = [1, 0] : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>

    // FFT
    %fft_result = stablehlo.fft %transposed, type = FFT, length = [2, 2] : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>

    // Reshape to [4, 1]
    %reshaped = stablehlo.reshape %fft_result : (tensor<2x2xcomplex<f64>>) -> tensor<4x1xcomplex<f64>>

    // dot_general(A, reshaped)
    %y = stablehlo.dot_general %A, %reshaped, contracting_dims = [1] x [0]
        : (tensor<2x4xcomplex<f64>>, tensor<4x1xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>

    %real_y = stablehlo.real %y : (tensor<2x1xcomplex<f64>>) -> tensor<2x1xf64>
    %slice = "stablehlo.slice"(%real_y) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 1, 1>, strides = array<i64: 1, 1>} : (tensor<2x1xf64>) -> tensor<1x1xf64>
    %result = stablehlo.reshape %slice : (tensor<1x1xf64>) -> tensor<f64>

    return %x#0, %result : tensor<2xui64>, tensor<f64>
  }
}
