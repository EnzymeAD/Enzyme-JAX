// RUN: enzymexlamlir-opt %s --inline-infer-regions --sicm | FileCheck %s
// RUN: enzymexlamlir-opt %s --inline-infer-regions --sicm --expand-impulse | FileCheck %s --check-prefix=LOOP

// Both solves are replaced by W = Q diag(1/sqrt(d)) Q^T, W = O L^{-1} with O
// orthogonal. The consumer check follows L^{-1} X through the product with
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
    %w = "stablehlo.triangular_solve"(%L, %ycol) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
    %X = stablehlo.slice %C [0:3, 0:2] : (tensor<3x3xf64>) -> tensor<3x2xf64>
    %v = "stablehlo.triangular_solve"(%L, %X) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
    %beta = stablehlo.constant dense<[0.5, -1.5]> : tensor<2xf64>
    %vb = stablehlo.dot_general %v, %beta, contracting_dims = [1] x [0] : (tensor<3x2xf64>, tensor<2xf64>) -> tensor<3xf64>
    %vbcol = stablehlo.reshape %vb : (tensor<3xf64>) -> tensor<3x1xf64>
    %r = stablehlo.subtract %w, %vbcol : tensor<3x1xf64>
    %q = stablehlo.dot_general %r, %r, contracting_dims = [0, 1] x [0, 1] : (tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<f64>
    %mhalf = stablehlo.constant dense<-0.5> : tensor<f64>
    %quad = stablehlo.multiply %mhalf, %q : tensor<f64>
    %d = stablehlo.dot_general %L, %I, batching_dims = [1] x [1], contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3xf64>
    %logd = stablehlo.log %d : tensor<3xf64>
    %half_logdet = stablehlo.reduce(%logd init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %lp = stablehlo.subtract %quad, %half_logdet : tensor<f64>
    return %lp : tensor<f64>
  }

  func.func @model(%rng : tensor<2xui64>, %C : tensor<3x3xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %s:2 = impulse.sample @halfnormal_sampler(%rng, %std) {
      logpdf = @halfnormal_logpdf,
      symbol = #impulse.symbol<1>,
      support = #impulse.support<POSITIVE>
    } : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = impulse.sample @halfnormal_sampler(%s#0, %std) {
      logpdf = @halfnormal_logpdf,
      symbol = #impulse.symbol<2>,
      support = #impulse.support<POSITIVE>
    } : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %y:2 = impulse.sample @y_sampler(%t#0, %s#1, %t#1, %C) {
      logpdf = @y_logpdf,
      symbol = #impulse.symbol<3>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  // CHECK-LABEL: func.func @main
  // CHECK-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[C:[^:]+]]: tensor<3x3xf64>, %{{[^:]+}}: tensor<1x5xf64>)
  // CHECK: %[[Q:[^,]+]], %[[LAMBDA:[^,]+]], %{{[^ ]+}} = enzymexla.lapack.syevd %[[C]] {uplo = #enzymexla.uplo<L>} : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xf64>, tensor<i64>)
  // CHECK-NEXT: %[[QT:.+]] = stablehlo.transpose %[[Q]], dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[YCOL:.+]] = stablehlo.reshape %{{.+}} : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[QTY:.+]] = stablehlo.dot_general %[[QT]], %[[YCOL]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[X:.+]] = stablehlo.slice %[[C]] [0:3, 0:2] : (tensor<3x3xf64>) -> tensor<3x2xf64>
  // CHECK-NEXT: %[[QTX:.+]] = stablehlo.dot_general %[[QT]], %[[X]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
  // CHECK-NEXT: impulse.infer_region @model
  // CHECK-NOT: stablehlo.cholesky
  // CHECK: impulse.yield %{{.+}}#0, %{{.+}}#1 : tensor<2xui64>, tensor<3xf64>
  // CHECK-NEXT: } logpdf {
  // CHECK-NEXT: ^bb0(%[[S:.+]]: tensor<f64>, %[[T:.+]]: tensor<f64>):
  // CHECK-NEXT: %[[LPS:.+]] = stablehlo.negate %[[S]] : tensor<f64>
  // CHECK-NEXT: %[[LPT:.+]] = stablehlo.negate %[[T]] : tensor<f64>
  // CHECK-NEXT: %[[LPST:.+]] = arith.addf %[[LPS]], %[[LPT]] : tensor<f64>
  // CHECK-NEXT: %[[SB:.+]] = stablehlo.broadcast_in_dim %[[S]], dims = [] : (tensor<f64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[SL:.+]] = stablehlo.multiply %[[SB]], %[[LAMBDA]] : tensor<3xf64>
  // CHECK-NEXT: %[[TB:.+]] = stablehlo.broadcast_in_dim %[[T]], dims = [] : (tensor<f64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[D:.+]] = stablehlo.add %[[SL]], %[[TB]] : tensor<3xf64>
  // L^{-1} y -> Q diag(1/sqrt(d)) Q^T y
  // CHECK-NEXT: %[[SQD:.+]] = stablehlo.sqrt %[[D]] : tensor<3xf64>
  // CHECK-NEXT: %[[ISQD:.+]] = stablehlo.divide %{{.+}}, %[[SQD]] : tensor<3xf64>
  // CHECK-NEXT: %[[B1:.+]] = stablehlo.broadcast_in_dim %[[ISQD]], dims = [0] : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[M1:.+]] = stablehlo.multiply %[[B1]], %[[QTY]] : tensor<3x1xf64>
  // CHECK-NEXT: %[[W:.+]] = stablehlo.dot_general %[[Q]], %[[M1]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[SQDD:.+]] = stablehlo.sqrt %[[D]] : tensor<3xf64>
  // L^{-1} X -> Q diag(1/sqrt(d)) Q^T X
  // CHECK-NEXT: %[[B2:.+]] = stablehlo.broadcast_in_dim %[[ISQD]], dims = [0] : (tensor<3xf64>) -> tensor<3x2xf64>
  // CHECK-NEXT: %[[M2:.+]] = stablehlo.multiply %[[B2]], %[[QTX]] : tensor<3x2xf64>
  // CHECK-NEXT: %[[V:.+]] = stablehlo.dot_general %[[Q]], %[[M2]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
  // CHECK-NEXT: %[[VB:.+]] = stablehlo.dot_general %[[V]], %{{.+}}, contracting_dims = [1] x [0] : (tensor<3x2xf64>, tensor<2xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[VBCOL:.+]] = stablehlo.reshape %[[VB]] : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[R:.+]] = stablehlo.subtract %[[W]], %[[VBCOL]] : tensor<3x1xf64>
  // CHECK-NEXT: %[[NORM:.+]] = stablehlo.dot_general %[[R]], %[[R]], contracting_dims = [0, 1] x [0, 1] : (tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<f64>
  // CHECK-NEXT: %{{.+}} = stablehlo.multiply %{{.+}}, %[[NORM]] : tensor<f64>
  // CHECK-NEXT: %[[LOGD:.+]] = stablehlo.log %[[SQDD]] : tensor<3xf64>
  // CHECK-NEXT: %{{.+}} = stablehlo.reduce(%[[LOGD]] init: %{{.+}}) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>

  // LOOP-LABEL: func.func @main
  // LOOP-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[C:[^:]+]]: tensor<3x3xf64>, %{{[^:]+}}: tensor<1x5xf64>)
  // LOOP-NOT: stablehlo.cholesky
  // LOOP: %{{[^,]+}}, %[[LAMBDA:[^,]+]], %{{[^ ]+}} = enzymexla.lapack.syevd %[[C]]
  // LOOP-NOT: stablehlo.cholesky
  // LOOP: impulse.for
  // LOOP-NOT: stablehlo.cholesky
  // LOOP: impulse.for
  // LOOP-NOT: stablehlo.cholesky
  // LOOP: stablehlo.multiply %{{.+}}, %[[LAMBDA]] : tensor<3xf64>
  // LOOP-NOT: stablehlo.cholesky
  func.func @main(%rng : tensor<2xui64>, %C : tensor<3x3xf64>,
                  %trace : tensor<1x5xf64>)
      -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %result:4 = impulse.infer @model(%rng, %C) given %trace
        init_step_size = %step_size {
      selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x5xf64>, tensor<f64>)
        -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %result#0, %result#1, %result#2, %result#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}
