// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: DotAbsorbFFT with contracting dim 0
// ============================================================================
//
// Model: dot_general(A:[6,5], reshape(fft(X:[2,3])), contracting_dims=[0]x[0])
//        where A is sample-invariant.
// When contractDim=0, the FFT dims are at the front after reshape, so SICM
// inserts transposes to move them to trailing position for StableHLO FFT:
//   A[6,5] → reshape[2,3,5] → transpose[5,2,3] → fft → transpose[2,3,5] → reshape[6,5]

// CHECK-LABEL: func.func @test_dot_absorb_fft_dim0
// CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[PRIOR:.*]]: tensor<f64>, %[[A:.*]]: tensor<6x5xcomplex<f64>>, %[[TRACE:.*]]: tensor<1x1xf64>
// FFT of A hoisted with transpose sandwich
// CHECK: stablehlo.reshape %[[A]] : (tensor<6x5xcomplex<f64>>) -> tensor<2x3x5xcomplex<f64>>
// CHECK-NEXT: stablehlo.transpose
// CHECK-NEXT: stablehlo.fft
// CHECK-NEXT: stablehlo.transpose
// CHECK-NEXT: stablehlo.reshape
// CHECK-NOT: stablehlo.fft
// CHECK: enzyme.mcmc_region
// Inside: no fft, dot_general uses pre-FFT'd matrix
// CHECK: enzyme.sample_region
// CHECK-NOT: stablehlo.fft
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

  func.func @test_dot_absorb_fft_dim0(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                       %A : tensor<6x5xcomplex<f64>>,
                                       %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:3 = enzyme.mcmc @model_fft_dim0(%rng, %prior, %A) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<6x5xcomplex<f64>>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_fft_dim0(%rng : tensor<2xui64>, %prior : tensor<f64>,
                              %A : tensor<6x5xcomplex<f64>>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %x:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Build a 2x3 complex matrix from the sample
    %x_complex = stablehlo.complex %x#1, %x#1 : tensor<complex<f64>>
    %x_mat = stablehlo.broadcast_in_dim %x_complex, dims = [] : (tensor<complex<f64>>) -> tensor<2x3xcomplex<f64>>

    // FFT
    %fft_result = stablehlo.fft %x_mat, type = FFT, length = [2, 3] : (tensor<2x3xcomplex<f64>>) -> tensor<2x3xcomplex<f64>>

    // Reshape to [6] for dot
    %reshaped = stablehlo.reshape %fft_result : (tensor<2x3xcomplex<f64>>) -> tensor<6xcomplex<f64>>

    // dot_general with contracting dim 0 on LHS
    %y = stablehlo.dot_general %A, %reshaped, contracting_dims = [0] x [0]
        : (tensor<6x5xcomplex<f64>>, tensor<6xcomplex<f64>>) -> tensor<5xcomplex<f64>>

    // Extract real part of first element as scalar
    %real_y = stablehlo.real %y : (tensor<5xcomplex<f64>>) -> tensor<5xf64>
    %slice = "stablehlo.slice"(%real_y) {start_indices = array<i64: 0>, limit_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<5xf64>) -> tensor<1xf64>
    %result = stablehlo.reshape %slice : (tensor<1xf64>) -> tensor<f64>

    return %x#0, %result : tensor<2xui64>, tensor<f64>
  }
}
