// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: Comrade NUFFT chain — mini version of the EHT black hole imaging model
// ============================================================================
//
// Hot path (sample-dependent):
//   convert(x_real:[K]) → complex → multiply(·, weights:[K]) →
//   scatter_real/scatter_imag(zeros:[N1,N2], idx:[K,2], ·) → complex →
//   transpose([1,0]) → fft(FFT,[N1,N2]) → reshape([P,1]) →
//   dot_general(A:[M,P], ·)
//
// Here: M=3 (visibility stations), K=2 (pixels), N1=N2=2, P=N1*N2=4
//
// Expected SICM fixpoint behavior:
//   Iter 1: DotAbsorbFFT absorbs FFT into A   → A' = fft(reshape(A))
//   Iter 2: DotAbsorbTranspose absorbs transpose → A'' = transpose(A')
//   Iter 3: DotAbsorbScatter absorbs scatter   → A''' = gather(A'', idx)
//
// After all 3 iterations, the dot_general shrinks from [M,P]×[P,1] to [M,K]×[K,1]
// and no FFT, transpose, or scatter ops remain in the sample-dependent path.

// CHECK-LABEL: func.func @test_comrade_chain
// CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[PRIOR:.*]]: tensor<f64>,
// CHECK-SAME: %[[A:.*]]: tensor<3x4xcomplex<f64>>,
// CHECK-SAME: %[[WEIGHTS:.*]]: tensor<2xcomplex<f64>>,
// CHECK-SAME: %[[IDX:.*]]: tensor<2x2xi32>,
// CHECK-SAME: %[[TRACE:.*]]: tensor<1x1xf64>

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
// CHECK: enzyme.sample_region
// CHECK-NOT: stablehlo.fft
// CHECK-NOT: stablehlo.transpose
// CHECK-NOT: stablehlo.scatter
// The dot_general should use the gathered (smaller) matrix
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

  func.func @test_comrade_chain(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                 %A : tensor<3x4xcomplex<f64>>,
                                 %weights : tensor<2xcomplex<f64>>,
                                 %idx : tensor<2x2xi32>,
                                 %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:3 = enzyme.mcmc @model_comrade(%rng, %prior, %A, %weights, %idx) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<3x4xcomplex<f64>>, tensor<2xcomplex<f64>>, tensor<2x2xi32>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // Mini Comrade model:
  //   x ~ Normal(prior, 1)
  //   x_vec = broadcast(x) : [2]
  //   x_complex = convert(x_vec) : complex
  //   weighted = multiply(x_complex, weights)    -- weights is invariant
  //   re = real(weighted), im = imag(weighted)
  //   scatter_re = scatter(zeros[2,2], idx, re)  -- idx is invariant
  //   scatter_im = scatter(zeros[2,2], idx, im)
  //   grid = complex(scatter_re, scatter_im)
  //   transposed = transpose(grid, [1,0])
  //   fft_out = fft(transposed, FFT, [2,2])
  //   reshaped = reshape(fft_out) : [4,1]
  //   vis = dot_general(A, reshaped)             -- A is invariant [3,4]
  //   return real(vis)[0]
  func.func @model_comrade(%rng : tensor<2xui64>, %prior : tensor<f64>,
                            %A : tensor<3x4xcomplex<f64>>,
                            %weights : tensor<2xcomplex<f64>>,
                            %idx : tensor<2x2xi32>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %x:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Broadcast sample to pixel vector
    %x_vec = stablehlo.broadcast_in_dim %x#1, dims = [] : (tensor<f64>) -> tensor<2xf64>

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

    return %x#0, %result : tensor<2xui64>, tensor<f64>
  }
}
