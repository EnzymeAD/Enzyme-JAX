// RUN: enzymexlamlir-opt %s --inline-infer-regions --sicm | FileCheck %s
// RUN: enzymexlamlir-opt %s --inline-infer-regions --sicm --expand-impulse | FileCheck %s --check-prefix=LOOP

// L^{-1} (L^{-T} y) is (L^T L)^{-1} y, not K^{-1} y, so it keeps the factor.

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

  func.func private @y_sampler(%rng : tensor<2xui64>, %s : tensor<f64>,
                               %t : tensor<f64>, %C : tensor<3x3xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %zero = stablehlo.constant dense<0.0> : tensor<3xf64>
    return %rng, %zero : tensor<2xui64>, tensor<3xf64>
  }

  func.func private @y_logpdf(%y : tensor<3xf64>, %s : tensor<f64>,
                              %t : tensor<f64>, %C : tensor<3x3xf64>)
      -> tensor<f64> {
    %I = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %sb = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %sC = stablehlo.multiply %sb, %C : tensor<3x3xf64>
    %tb = stablehlo.broadcast_in_dim %t, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %tI = stablehlo.multiply %tb, %I : tensor<3x3xf64>
    %K = stablehlo.add %sC, %tI : tensor<3x3xf64>
    %L = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>
    %ycol = stablehlo.reshape %y : (tensor<3xf64>) -> tensor<3x1xf64>
    %a1 = "stablehlo.triangular_solve"(%L, %ycol) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
    %a = "stablehlo.triangular_solve"(%L, %a1) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
    %qa = stablehlo.dot_general %ycol, %a, contracting_dims = [0, 1] x [0, 1] : (tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<f64>
    %b1 = "stablehlo.triangular_solve"(%L, %ycol) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
    %b = "stablehlo.triangular_solve"(%L, %b1) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
    %qb = stablehlo.dot_general %ycol, %b, contracting_dims = [0, 1] x [0, 1] : (tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<f64>
    %lp = stablehlo.add %qa, %qb : tensor<f64>
    return %lp : tensor<f64>
  }

  func.func @model(%rng : tensor<2xui64>, %C : tensor<3x3xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %s:2 = impulse.sample @halfnormal_sampler(%rng, %std) <{
      logpdf = @halfnormal_logpdf,
      symbol = #impulse.symbol<1>,
      support = #impulse.support<POSITIVE>
    }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = impulse.sample @halfnormal_sampler(%s#0, %std) <{
      logpdf = @halfnormal_logpdf,
      symbol = #impulse.symbol<2>,
      support = #impulse.support<POSITIVE>
    }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %y:2 = impulse.sample @y_sampler(%t#0, %s#1, %t#1, %C) <{
      logpdf = @y_logpdf,
      symbol = #impulse.symbol<3>
    }> : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  // CHECK-LABEL: func.func @main
  // CHECK-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[C:[^:]+]]: tensor<3x3xf64>, %{{[^:]+}}: tensor<1x5xf64>)
  // CHECK: %[[Q:[^,]+]], %[[LAMBDA:[^,]+]], %{{[^ ]+}} = enzymexla.lapack.syevd %[[C]] <uplo = L> : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xf64>, tensor<i64>)
  // CHECK-NEXT: %[[QT:.+]] = stablehlo.transpose %[[Q]], dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[YCOL:.+]] = stablehlo.reshape %{{.+}} : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[QTY:.+]] = stablehlo.dot_general %[[QT]], %[[YCOL]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: impulse.infer_region @model
  // CHECK: impulse.yield %{{.+}}#0, %{{.+}}#1 : tensor<2xui64>, tensor<3xf64>
  // CHECK-NEXT: } logpdf {
  // CHECK-NEXT: ^bb0(%[[S:.+]]: tensor<f64>, %[[T:.+]]: tensor<f64>):
  // CHECK-NEXT: %[[LPS:.+]] = stablehlo.negate %[[S]] : tensor<f64>
  // CHECK-NEXT: %[[LPT:.+]] = stablehlo.negate %[[T]] : tensor<f64>
  // CHECK-NEXT: %[[LPST:.+]] = arith.addf %[[LPS]], %[[LPT]] : tensor<f64>
  // CHECK-NEXT: %[[SBM:.+]] = stablehlo.broadcast_in_dim %[[S]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[SC:.+]] = stablehlo.multiply %[[SBM]], %[[C]] : tensor<3x3xf64>
  // CHECK-NEXT: %[[TBM:.+]] = stablehlo.broadcast_in_dim %[[T]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[TI:.+]] = stablehlo.multiply %[[TBM]], %{{.+}} : tensor<3x3xf64>
  // CHECK-NEXT: %[[K:.+]] = stablehlo.add %[[SC]], %[[TI]] : tensor<3x3xf64>
  // CHECK-NEXT: %[[SB:.+]] = stablehlo.broadcast_in_dim %[[S]], dims = [] : (tensor<f64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[SL:.+]] = stablehlo.multiply %[[SB]], %[[LAMBDA]] : tensor<3xf64>
  // CHECK-NEXT: %[[TB:.+]] = stablehlo.broadcast_in_dim %[[T]], dims = [] : (tensor<f64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[D:.+]] = stablehlo.add %[[SL]], %[[TB]] : tensor<3xf64>
  // CHECK-NEXT: %[[SQD:.+]] = stablehlo.sqrt %[[D]] : tensor<3xf64>
  // CHECK-NEXT: %[[ISQD:.+]] = stablehlo.divide %{{.+}}, %[[SQD]] : tensor<3xf64>
  // CHECK-NEXT: %[[L:.+]] = stablehlo.cholesky %[[K]], lower = true : tensor<3x3xf64>
  // CHECK-NEXT: %[[B:.+]] = stablehlo.broadcast_in_dim %[[ISQD]], dims = [0] : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[U:.+]] = stablehlo.multiply %[[B]], %[[QTY]] : tensor<3x1xf64>
  // CHECK-NEXT: %{{.+}} = stablehlo.dot_general %[[U]], %[[U]], contracting_dims = [0, 1] x [0, 1] : (tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[IN:.+]] = "stablehlo.triangular_solve"(%[[L]], %[[YCOL]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[OUT:.+]] = "stablehlo.triangular_solve"(%[[L]], %[[IN]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %{{.+}} = stablehlo.dot_general %[[YCOL]], %[[OUT]], contracting_dims = [0, 1] x [0, 1] : (tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<f64>

  // LOOP-LABEL: func.func @main
  // LOOP-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[C:[^:]+]]: tensor<3x3xf64>, %{{[^:]+}}: tensor<1x5xf64>)
  // LOOP: %{{[^,]+}}, %[[LAMBDA:[^,]+]], %{{[^ ]+}} = enzymexla.lapack.syevd %[[C]]
  // LOOP: impulse.for
  // LOOP: impulse.for
  // LOOP: stablehlo.multiply %{{.+}}, %[[LAMBDA]] : tensor<3xf64>
  // LOOP: %[[L:.+]] = stablehlo.cholesky %{{.+}}, lower = true : tensor<3x3xf64>
  // LOOP: "stablehlo.triangular_solve"(%[[L]], %{{.+}}) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}>
  func.func @main(%rng : tensor<2xui64>, %C : tensor<3x3xf64>,
                  %trace : tensor<1x5xf64>)
      -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %result:4 = impulse.infer @model(%rng, %C) given %trace
        init_step_size = %step_size <{
      selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    }> : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x5xf64>, tensor<f64>)
        -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %result#0, %result#1, %result#2, %result#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}
