// RUN: enzymexlamlir-opt %s --inline-infer-regions --sicm="eigenlift-guard=force-lift guard-versioning=false" | FileCheck %s --check-prefix=LIFT
// RUN: enzymexlamlir-opt %s --inline-infer-regions --sicm="eigenlift-guard=force-fallback guard-versioning=false" | FileCheck %s --check-prefix=FALLBACK

// force-fallback ands the check with x != x, so the original cholesky runs.

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
                               %t : tensor<f64>, %C : tensor<3x3xf64>, %X : tensor<3x3xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %zero = stablehlo.constant dense<0.0> : tensor<3xf64>
    return %rng, %zero : tensor<2xui64>, tensor<3xf64>
  }

  func.func private @y_logpdf(%y : tensor<3xf64>, %s : tensor<f64>,
                              %t : tensor<f64>, %C : tensor<3x3xf64>, %X : tensor<3x3xf64>)
      -> tensor<f64> {
    %I = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %sb = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %sC = stablehlo.multiply %sb, %C : tensor<3x3xf64>
    %tb = stablehlo.broadcast_in_dim %t, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %tX = stablehlo.multiply %tb, %X : tensor<3x3xf64>
    %K = stablehlo.add %sC, %tX : tensor<3x3xf64>
    %L = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>
    %d = stablehlo.dot_general %L, %I, batching_dims = [1] x [1], contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3xf64>
    %logd = stablehlo.log %d : tensor<3xf64>
    %half_logdet = stablehlo.reduce(%logd init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %neg = stablehlo.negate %half_logdet : tensor<f64>
    return %neg : tensor<f64>
  }

  func.func @model(%rng : tensor<2xui64>, %C : tensor<3x3xf64>, %X : tensor<3x3xf64>)
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
    %y:2 = impulse.sample @y_sampler(%t#0, %s#1, %t#1, %C, %X) <{
      logpdf = @y_logpdf,
      symbol = #impulse.symbol<3>
    }> : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<3x3xf64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  func.func @main(%rng : tensor<2xui64>, %C : tensor<3x3xf64>, %X : tensor<3x3xf64>,
                  %trace : tensor<1x5xf64>)
      -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %result:4 = impulse.infer @model(%rng, %C, %X) given %trace
        init_step_size = %step_size <{
      selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    }> : (tensor<2xui64>, tensor<3x3xf64>, tensor<3x3xf64>, tensor<1x5xf64>, tensor<f64>)
        -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %result#0, %result#1, %result#2, %result#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// LIFT-LABEL: func.func @main
// LIFT-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[C:[^:]+]]: tensor<3x3xf64>, %[[D:[^:]+]]: tensor<3x3xf64>, %{{[^:]+}}: tensor<1x5xf64>)
// LIFT-NOT: enzymexla.lapack.potrf
// LIFT: %[[LB:.+]] = stablehlo.cholesky %[[D]], lower = true : tensor<3x3xf64>
// LIFT: enzymexla.lapack.syevd
// LIFT: } logpdf {
// LIFT-NOT: stablehlo.if
// LIFT-NOT: stablehlo.cholesky
// LIFT: impulse.yield

// FALLBACK-LABEL: func.func @main
// FALLBACK-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[C:[^:]+]]: tensor<3x3xf64>, %[[D:[^:]+]]: tensor<3x3xf64>, %{{[^:]+}}: tensor<1x5xf64>)
// FALLBACK: %{{.+}}, %[[INFO:.+]] = enzymexla.lapack.potrf %[[D]]
// FALLBACK-NEXT: %[[OK:.+]] = stablehlo.compare EQ, %[[INFO]], %{{.+}} : (tensor<i64>, tensor<i64>) -> tensor<i1>
// FALLBACK: %[[X0:.+]] = stablehlo.slice %[[D]] [0:1, 0:1] : (tensor<3x3xf64>) -> tensor<1x1xf64>
// FALLBACK-NEXT: %[[X:.+]] = stablehlo.reshape %[[X0]] : (tensor<1x1xf64>) -> tensor<f64>
// FALLBACK-NEXT: %[[NAN:.+]] = stablehlo.compare NE, %[[X]], %[[X]] : (tensor<f64>, tensor<f64>) -> tensor<i1>
// FALLBACK-NEXT: %[[P:.+]] = stablehlo.and %[[OK]], %[[NAN]] : tensor<i1>
// FALLBACK: } logpdf {
// FALLBACK: "stablehlo.if"(%[[P]])
