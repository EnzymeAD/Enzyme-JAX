// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(inline-infer-regions,canonicalize,cse,lower-impulse-to-stablehlo{math-twins-only=true},arith-raise{stablehlo=true},canonicalize,cse,sicm)' | FileCheck %s --check-prefix=REGION
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(inline-infer-regions,canonicalize,cse,lower-impulse-to-stablehlo{math-twins-only=true},arith-raise{stablehlo=true},canonicalize,cse,sicm,canonicalize,cse,expand-impulse,lower-impulse-to-stablehlo{backend=cpu},outline-enzyme-regions,enzyme,canonicalize,remove-unnecessary-enzyme-ops,canonicalize,cse)' | FileCheck %s --check-prefix=GRAD

// sigma_sq * R: every per-step n x n op reduces to one solve against chol(R).

module {
  func.func private @halfnormal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %sample = stablehlo.abs %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }
  func.func private @halfnormal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }
  func.func private @normal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %std : tensor<2xui64>, tensor<f64>
  }
  func.func private @normal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %sq = stablehlo.multiply %x, %x : tensor<f64>
    %neg = stablehlo.negate %sq : tensor<f64>
    return %neg : tensor<f64>
  }
  func.func @model(%rng : tensor<2xui64>, %R_data : tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>) {
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %sigsq:2 = impulse.sample @halfnormal_sampler(%rng, %std) {
      logpdf = @halfnormal_logpdf, symbol = #impulse.symbol<1>, support = #impulse.support<POSITIVE>
    } : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %m:2 = impulse.sample @normal_sampler(%sigsq#0, %std) {
      logpdf = @normal_logpdf, symbol = #impulse.symbol<2>, support = #impulse.support<REAL>
    } : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %sigsq_mat = stablehlo.broadcast_in_dim %sigsq#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %K = stablehlo.multiply %R_data, %sigsq_mat : tensor<3x3xf64>
    %mu = stablehlo.broadcast_in_dim %m#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %y:2 = "impulse.sample"(%m#0, %mu, %K) {
      dist = #impulse.dist<MVN>, symbol = #impulse.symbol<3>, support = #impulse.support<REAL>
    } : (tensor<2xui64>, tensor<3xf64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  // REGION-LABEL: func.func @test
  // REGION-DAG: %[[I:.+]] = stablehlo.constant dense<{{\[\[}}1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<3x3xf64>
  // REGION-DAG: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf64>
  // REGION-DAG: %[[ONE:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
  // REGION: %[[CHOL:.+]] = stablehlo.cholesky %arg1 : tensor<3x3xf64>
  // REGION-NEXT: %[[MASKED:.+]] = stablehlo.multiply %[[CHOL]], %[[I]] : tensor<3x3xf64>
  // REGION-NEXT: %[[DIAG:.+]] = stablehlo.dot_general %[[MASKED]], %[[ONES]], contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
  // REGION-NEXT: %[[LOGDIAG:.+]] = stablehlo.log %[[DIAG]] : tensor<3xf64>
  // REGION-NEXT: impulse.infer_region @model
  // REGION: ^bb0(%[[SIGSQ:[^:]+]]: tensor<f64>, %[[M:[^:]+]]: tensor<f64>):
  // REGION-NOT: tensor<3x3xf64>
  // REGION: %[[MU:.+]] = stablehlo.broadcast_in_dim %[[M]], dims = [] : (tensor<f64>) -> tensor<3xf64>
  // REGION-NEXT: %[[DIFF:.+]] = stablehlo.subtract %{{.+}}, %[[MU]] : tensor<3xf64>
  // REGION-NEXT: %[[SQRT:.+]] = stablehlo.sqrt %[[SIGSQ]] {{.*}}: tensor<f64>
  // REGION-NEXT: %[[DCOL:.+]] = impulse.reshape %[[DIFF]] : (tensor<3xf64>) -> tensor<3x1xf64>
  // REGION-NEXT: %[[W:.+]] = "stablehlo.triangular_solve"(%[[CHOL]], %[[DCOL]]) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
  // REGION-NEXT: %[[INV:.+]] = stablehlo.divide %[[ONE]], %[[SQRT]] : tensor<f64>
  // REGION-NEXT: %[[INVB:.+]] = stablehlo.broadcast_in_dim %[[INV]], dims = [] : (tensor<f64>) -> tensor<3x1xf64>
  // REGION-NEXT: %[[WS:.+]] = stablehlo.multiply %[[INVB]], %[[W]] : tensor<3x1xf64>
  // REGION-NEXT: %[[WV:.+]] = impulse.reshape %[[WS]] : (tensor<3x1xf64>) -> tensor<3xf64>
  // REGION-NEXT: %[[WSQ:.+]] = stablehlo.multiply %[[WV]], %[[WV]] : tensor<3xf64>
  // REGION-NEXT: %{{.+}} = stablehlo.dot_general %[[WSQ]], %[[ONES]], contracting_dims = [0] x [0] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
  // REGION-NOT: tensor<3x3xf64>
  // REGION: %[[LOGS:.+]] = stablehlo.log %[[SQRT]] : tensor<f64>
  // REGION-NEXT: %[[LOGSB:.+]] = stablehlo.broadcast_in_dim %[[LOGS]], dims = [] : (tensor<f64>) -> tensor<3xf64>
  // REGION-NEXT: %{{.+}} = stablehlo.add %[[LOGSB]], %[[LOGDIAG]] : tensor<3xf64>
  // REGION-NOT: tensor<3x3xf64>
  // REGION: impulse.yield

  // GRAD-LABEL: func.func private @diffetest_to_diff1(
  // GRAD-SAME: %[[CHOLARG:[^:]+]]: tensor<3x3xf64>
  // GRAD-NOT: tensor<3x3xf64>
  // GRAD: %[[SQRT:.+]] = stablehlo.sqrt %{{.+}} : tensor<f64>
  // GRAD-NOT: tensor<3x3xf64>
  // GRAD: "stablehlo.triangular_solve"(%[[CHOLARG]], %{{.+}}) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
  // GRAD-NOT: tensor<3x3xf64>
  // GRAD: stablehlo.reduce(%{{.+}} init: %{{.+}}) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
  // GRAD-NOT: tensor<3x3xf64>
  // GRAD: stablehlo.divide %{{.+}}, %[[SQRT]] : tensor<f64>
  // GRAD-NOT: tensor<3x3xf64>
  // GRAD: stablehlo.reduce(%{{.+}} init: %{{.+}}) applies stablehlo.add across dimensions = [0, 1] : (tensor<3x1xf64>, tensor<f64>) -> tensor<f64>
  // GRAD-NOT: tensor<3x3xf64>
  // GRAD: stablehlo.divide %{{.+}}, %[[SQRT]] : tensor<f64>
  // GRAD-NOT: tensor<3x3xf64>
  // GRAD: "stablehlo.triangular_solve"(%[[CHOLARG]], %{{.+}}) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
  // GRAD-NOT: tensor<3x3xf64>
  // GRAD: return
  func.func @test(%rng : tensor<2xui64>, %R_data : tensor<3x3xf64>, %trace : tensor<1x5xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %result:4 = impulse.infer @model(%rng, %R_data) given %trace init_step_size = %step_size {
      selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64, thinning = 1 : i64, num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x5xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %result#0, %result#1, %result#2, %result#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}
