// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(inline-infer-regions,canonicalize,cse,lower-impulse-to-stablehlo{math-twins-only=true},arith-raise{stablehlo=true},canonicalize,cse,sicm)' | FileCheck %s
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(inline-infer-regions,canonicalize,cse,lower-impulse-to-stablehlo{math-twins-only=true},arith-raise{stablehlo=true},canonicalize,cse,sicm,expand-impulse)' | FileCheck %s --check-prefix=LOOP

// Emitted MVN density: U is used only by the whitening solve and the log det.

module {
  func.func private @halfnormal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %sample = stablehlo.abs %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }
  func.func private @halfnormal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }
  func.func @model(%rng : tensor<2xui64>, %C : tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>) {
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %I = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    %s:2 = impulse.sample @halfnormal_sampler(%rng, %std) {
      logpdf = @halfnormal_logpdf, symbol = #impulse.symbol<1>, support = #impulse.support<POSITIVE>
    } : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = impulse.sample @halfnormal_sampler(%s#0, %std) {
      logpdf = @halfnormal_logpdf, symbol = #impulse.symbol<2>, support = #impulse.support<POSITIVE>
    } : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %sb = stablehlo.broadcast_in_dim %s#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %sC = stablehlo.multiply %sb, %C : tensor<3x3xf64>
    %tb = stablehlo.broadcast_in_dim %t#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %tI = stablehlo.multiply %tb, %I : tensor<3x3xf64>
    %K = stablehlo.add %sC, %tI : tensor<3x3xf64>
    %mu = stablehlo.constant dense<0.0> : tensor<3xf64>
    %y:2 = impulse.sample(%t#0, %mu, %K) {
      dist = #impulse.dist<MVN>, symbol = #impulse.symbol<3>, support = #impulse.support<REAL>
    } : (tensor<2xui64>, tensor<3xf64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }
  // CHECK-LABEL: func.func @main
  // CHECK-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[C:[^:]+]]: tensor<3x3xf64>, %{{[^:]+}}: tensor<1x5xf64>)
  // CHECK: %[[DIFF:[^ ]+]] = stablehlo.subtract %{{.+}}, %{{.+}} : tensor<3xf64>
  // CHECK-NEXT: %[[Q:[^,]+]], %[[LAMBDA:[^,]+]], %{{[^ ]+}} = enzymexla.lapack.syevd %[[C]] {uplo = #enzymexla.uplo<L>} : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xf64>, tensor<i64>)
  // CHECK-NEXT: %[[QT:.+]] = stablehlo.transpose %[[Q]], dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[DIFFCOL:.+]] = impulse.reshape %[[DIFF]] : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[QTB:.+]] = stablehlo.dot_general %[[QT]], %[[DIFFCOL]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: impulse.infer_region @model
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
  // CHECK-NEXT: %[[SQD:.+]] = stablehlo.sqrt %[[D]] : tensor<3xf64>
  // CHECK-NEXT: %[[ISQD:.+]] = stablehlo.divide %[[ONES:.+]], %[[SQD]] : tensor<3xf64>
  // CHECK-NEXT: %[[SQDD:.+]] = stablehlo.sqrt %[[D]] : tensor<3xf64>
  // CHECK-NEXT: %[[ISQDB:.+]] = stablehlo.broadcast_in_dim %[[ISQD]], dims = [0] : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[W:.+]] = stablehlo.multiply %[[ISQDB]], %[[QTB]] : tensor<3x1xf64>
  // CHECK-NEXT: %[[WV:.+]] = impulse.reshape %[[W]] : (tensor<3x1xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[WSQ:.+]] = stablehlo.multiply %[[WV]], %[[WV]] : tensor<3xf64>
  // CHECK-NEXT: %[[MAHA:.+]] = stablehlo.dot_general %[[WSQ]], %[[ONES]], contracting_dims = [0] x [0] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
  // CHECK-NEXT: %{{.+}} = stablehlo.multiply %[[MAHA]], %{{.+}} : tensor<f64>
  // CHECK-NEXT: %[[LOGD:.+]] = stablehlo.log %[[SQDD]] : tensor<3xf64>
  // CHECK-NEXT: %{{.+}} = stablehlo.dot_general %[[LOGD]], %[[ONES]], contracting_dims = [0] x [0] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>

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
  func.func @main(%rng : tensor<2xui64>, %C : tensor<3x3xf64>, %trace : tensor<1x5xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %result:4 = impulse.infer @model(%rng, %C) given %trace init_step_size = %step_size {
      selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64, thinning = 1 : i64, num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x5xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %result#0, %result#1, %result#2, %result#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}
