// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm --cse %s | FileCheck %s

// ============================================================================
// Test: CSE across sample sites via unified logpdf region
// ============================================================================
//
// Two sample sites share the same logpdf function which computes math.log(sigma).
// After SICM hoists sample-invariant ops and constructs the unified logpdf,
// math.log(sigma) is hoisted before the mcmc_region (sample-invariant), and
// both sites' logpdf contributions reference the single hoisted copy.

// CHECK-LABEL: func.func @test_flatten_cse
// math.log hoisted before mcmc_region (sample-invariant)
// CHECK: %[[LOG:.+]] = math.log
// CHECK: enzyme.mcmc_region
// Two sample_region ops with empty per-site logpdf bodies
// CHECK: enzyme.sample_region
// CHECK: enzyme.sample_region
// Unified logpdf region with 2 position args (both sites selected)
// CHECK: ^bb0(%[[X0:[^:]+]]: tensor<f64>, %[[X1:[^:]+]]: tensor<f64>):
// Site 0: normal logpdf using hoisted log
// CHECK-NEXT: arith.subf %[[X0]]
// CHECK-NEXT: arith.divf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.subf %{{.*}}, %[[LOG]]
// Site 1: same pattern, same hoisted log (CSE'd)
// CHECK-NEXT: arith.subf %[[X1]]
// CHECK-NEXT: arith.divf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.subf %{{.*}}, %[[LOG]]
// Sum contributions from both sites
// CHECK-NEXT: arith.addf
// CHECK-NEXT: enzyme.yield
// CHECK: num_position_args = 2

module {
  func.func private @normal_sampler(%rng : tensor<2xui64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %sample = arith.addf %mean, %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }

  func.func private @normal_logpdf(%x : tensor<f64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> tensor<f64> {
    // This math.log is the target for CSE: both sites compute it.
    %log_std = math.log %std : tensor<f64>
    %diff = arith.subf %x, %mean : tensor<f64>
    %scaled = arith.divf %diff, %std : tensor<f64>
    %sq = arith.mulf %scaled, %scaled : tensor<f64>
    %neg_half = stablehlo.constant dense<-0.5> : tensor<f64>
    %quad = arith.mulf %neg_half, %sq : tensor<f64>
    %lp = arith.subf %quad, %log_std : tensor<f64>
    return %lp : tensor<f64>
  }

  func.func @test_flatten_cse(
      %rng : tensor<2xui64>,
      %mu : tensor<f64>,
      %sigma : tensor<f64>,
      %trace : tensor<1x2xf64>) {

    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @test_flatten_cse_model(%rng, %mu, %sigma) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<0>], [#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<0>], [#enzyme.symbol<1>]],
      nuts_config = #enzyme.nuts_config<max_tree_depth = 10, max_delta_energy = 1000.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>, tensor<f64>)
        -> (tensor<1x2xf64>, tensor<1x5xf64>, tensor<2xui64>,
            tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>,
            tensor<f64>, tensor<1x2xf64>)
    return
  }

  func.func private @test_flatten_cse_model(%rng : tensor<2xui64>, %mu : tensor<f64>, %sigma : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>, tensor<f64>) {
    // Site 1: y1 ~ Normal(mu, sigma)
    %rng1, %y1 = enzyme.sample @normal_sampler(%rng, %mu, %sigma) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<0>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Site 2: y2 ~ Normal(mu, sigma) — same sigma, so logpdf shares math.log(sigma)
    %rng2, %y2 = enzyme.sample @normal_sampler(%rng1, %mu, %sigma) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    return %rng2, %y1, %y2 : tensor<2xui64>, tensor<f64>, tensor<f64>
  }
}
