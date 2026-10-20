// RUN: enzymexlamlir-opt %s --inline-infer-regions --sicm | FileCheck %s

module {
  func.func private @halfnormal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %sample = stablehlo.abs %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }

  func.func private @halfnormal_logpdf(%x : tensor<f64>, %std : tensor<f64>)
      -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  func.func private @mvn_sampler(%rng : tensor<2xui64>, %cov : tensor<3x3xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %zero = stablehlo.constant dense<0.0> : tensor<3xf64>
    return %rng, %zero : tensor<2xui64>, tensor<3xf64>
  }

  func.func private @mvn_logpdf(%x : tensor<3xf64>, %cov : tensor<3x3xf64>)
      -> tensor<f64> {
    %L = stablehlo.cholesky %cov, lower = true : tensor<3x3xf64>
    %x_col = stablehlo.reshape %x : (tensor<3xf64>) -> tensor<3x1xf64>
    %solved = "stablehlo.triangular_solve"(%L, %x_col) <{
      left_side = true, lower = true,
      unit_diagonal = false,
      transpose_a = #stablehlo<transpose NO_TRANSPOSE>
    }> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
    %log_L = stablehlo.log %L : tensor<3x3xf64>
    %sliced = stablehlo.slice %log_L [0:1, 0:1] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %result = stablehlo.reshape %sliced : (tensor<1x1xf64>) -> tensor<f64>
    return %result : tensor<f64>
  }

  func.func @model_scale_family_mvn(%rng : tensor<2xui64>,
                                     %R_data : tensor<3x3xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %tau:2 = impulse.sample @halfnormal_sampler(%rng, %std) <{
      logpdf = @halfnormal_logpdf,
      symbol = #impulse.symbol<1>,
      support = #impulse.support<POSITIVE>
    }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    %tau_sq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
    %tau_sq_mat = stablehlo.broadcast_in_dim %tau_sq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %K = stablehlo.multiply %tau_sq_mat, %R_data : tensor<3x3xf64>

    %y:2 = impulse.sample @mvn_sampler(%tau#0, %K) <{
      logpdf = @mvn_logpdf,
      symbol = #impulse.symbol<2>
    }> : (tensor<2xui64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  // CHECK-LABEL: func.func @test_scale_family_mvn
  // CHECK: %[[CHOL:.+]] = stablehlo.cholesky %{{.+}}, lower = true : tensor<3x3xf64>
  // CHECK-NEXT: %[[LOGCHOL:.+]] = stablehlo.log %[[CHOL]] : tensor<3x3xf64>
  // CHECK: impulse.infer_region @model_scale_family_mvn
  // CHECK: impulse.sample_region
  // CHECK: impulse.sample_region
  // CHECK: impulse.yield %{{.+}}#0, %{{.+}}#1 : tensor<2xui64>, tensor<3xf64>
  // CHECK-NEXT: } logpdf {
  // CHECK-NEXT: ^bb0(%[[TAU:.+]]: tensor<f64>):
  // CHECK-NEXT: %[[HN:.+]] = stablehlo.negate %[[TAU]] : tensor<f64>
  // CHECK-NEXT: %[[TAUSQ:.+]] = stablehlo.multiply %[[TAU]], %[[TAU]] {{.*}}: tensor<f64>
  // CHECK-NEXT: %[[SCALE:.+]] = stablehlo.sqrt %[[TAUSQ]] {{.*}}: tensor<f64>
  // CHECK-NEXT: %[[LOGSCALE:.+]] = stablehlo.log %[[SCALE]] : tensor<f64>
  // CHECK-NEXT: %[[LOGSCALEB:.+]] = stablehlo.broadcast_in_dim %[[LOGSCALE]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[LOGL:.+]] = stablehlo.add %[[LOGSCALEB]], %[[LOGCHOL]] : tensor<3x3xf64>
  // CHECK-NEXT: %[[SLICED:.+]] = stablehlo.slice %[[LOGL]] [0:1, 0:1] : (tensor<3x3xf64>) -> tensor<1x1xf64>
  // CHECK-NEXT: %[[MVN:.+]] = stablehlo.reshape %[[SLICED]] : (tensor<1x1xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[TOTAL:.+]] = arith.addf %[[HN]], %[[MVN]] : tensor<f64>
  // CHECK-NEXT: impulse.yield %[[TOTAL]], %[[HN]], %[[MVN]] : tensor<f64>, tensor<f64>, tensor<f64>
  // CHECK: num_position_args = 1
  // CHECK-SAME: position_size = 1
  func.func @test_scale_family_mvn(%rng : tensor<2xui64>,
                                    %R_data : tensor<3x3xf64>,
                                    %trace : tensor<1x4xf64>)
      -> (tensor<1x4xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %result:4 = impulse.infer @model_scale_family_mvn(%rng, %R_data) given %trace
        init_step_size = %step_size <{
      selection = [[#impulse.symbol<1>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    }> : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x4xf64>, tensor<f64>)
        -> (tensor<1x4xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %result#0, %result#1, %result#2, %result#3 : tensor<1x4xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}
