// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(inline-infer-regions,canonicalize,cse,lower-impulse-to-stablehlo{math-twins-only=true},arith-raise{stablehlo=true},canonicalize,cse,sicm)' | FileCheck %s

// Fully emitted MVN density: cholesky(R) still leaves the density region.

module {
  func.func private @halfnormal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %sample = stablehlo.abs %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }
  func.func private @halfnormal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }
  func.func @model(%rng : tensor<2xui64>, %R_data : tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>) {
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %tau:2 = impulse.sample @halfnormal_sampler(%rng, %std) <{
      logpdf = @halfnormal_logpdf, symbol = #impulse.symbol<1>, support = #impulse.support<POSITIVE>
    }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %tau_sq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
    %tau_sq_mat = stablehlo.broadcast_in_dim %tau_sq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %K = stablehlo.multiply %tau_sq_mat, %R_data : tensor<3x3xf64>
    %mu = stablehlo.constant dense<0.0> : tensor<3xf64>
    %y:2 = "impulse.sample"(%tau#0, %mu, %K) {
      dist = #impulse.dist<MVN>, symbol = #impulse.symbol<2>, support = #impulse.support<REAL>
    } : (tensor<2xui64>, tensor<3xf64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  // CHECK-LABEL: func.func @test
  // CHECK: %[[CHOL:.+]] = stablehlo.cholesky %arg1 : tensor<3x3xf64>
  // CHECK-NEXT: %[[DCOL:.+]] = impulse.reshape %{{.+}} : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[W:.+]] = "stablehlo.triangular_solve"(%[[CHOL]], %[[DCOL]]) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[MASKED:.+]] = stablehlo.multiply %[[CHOL]], %{{.+}} : tensor<3x3xf64>
  // CHECK-NEXT: %[[DIAG:.+]] = stablehlo.dot_general %[[MASKED]], %{{.+}}, contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[LOGDIAG:.+]] = stablehlo.log %[[DIAG]] : tensor<3xf64>
  // CHECK-NEXT: impulse.infer_region @model
  // CHECK: %[[TAUSQ:.+]] = stablehlo.multiply %[[TAUARG:.+]], %[[TAUARG]] {enzymexla.non_negative = {{.*}}} : tensor<f64>
  // CHECK-NEXT: %[[SQRT:.+]] = stablehlo.sqrt %[[TAUSQ]] {enzymexla.non_negative = {{.*}}} : tensor<f64>
  // CHECK-NEXT: %[[INV:.+]] = stablehlo.divide %{{.+}}, %[[SQRT]] : tensor<f64>
  // CHECK-NEXT: %[[INVB:.+]] = stablehlo.broadcast_in_dim %[[INV]], dims = [] : (tensor<f64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[WS:.+]] = stablehlo.multiply %[[INVB]], %[[W]] : tensor<3x1xf64>
  // CHECK-NEXT: %[[WV:.+]] = impulse.reshape %[[WS]] : (tensor<3x1xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[WSQ:.+]] = stablehlo.multiply %[[WV]], %[[WV]] : tensor<3xf64>
  // CHECK-NOT: tensor<3x3xf64>
  // CHECK: %[[LOGS:.+]] = stablehlo.log %[[SQRT]] : tensor<f64>
  // CHECK-NEXT: %[[LOGSB:.+]] = stablehlo.broadcast_in_dim %[[LOGS]], dims = [] : (tensor<f64>) -> tensor<3xf64>
  // CHECK-NEXT: %{{.+}} = stablehlo.add %[[LOGSB]], %[[LOGDIAG]] : tensor<3xf64>
  // CHECK-NOT: tensor<3x3xf64>
  // CHECK: impulse.yield
  func.func @test(%rng : tensor<2xui64>, %R_data : tensor<3x3xf64>, %trace : tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %result:4 = impulse.infer @model(%rng, %R_data) given %trace init_step_size = %step_size <{
      selection = [[#impulse.symbol<1>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64, thinning = 1 : i64, num_warmup = 0 : i64
    }> : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x4xf64>, tensor<f64>) -> (tensor<1x4xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %result#0, %result#1, %result#2, %result#3 : tensor<1x4xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}
