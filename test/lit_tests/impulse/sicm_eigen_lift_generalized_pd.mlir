// RUN: enzymexlamlir-opt %s --split-input-file --inline-infer-regions --sicm="guard-versioning=false" | FileCheck %s

// K = t*D + s*C, D a constant positive diagonal: fires with B = D.
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
    %B = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    %sb = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %sC = stablehlo.multiply %sb, %C : tensor<3x3xf64>
    %tb = stablehlo.broadcast_in_dim %t, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %tB = stablehlo.multiply %tb, %B : tensor<3x3xf64>
    %K = stablehlo.add %tB, %sC : tensor<3x3xf64>
    %L = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>
    %d = stablehlo.dot_general %L, %I, batching_dims = [1] x [1], contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3xf64>
    %logd = stablehlo.log %d : tensor<3xf64>
    %half_logdet = stablehlo.reduce(%logd init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %neg = stablehlo.negate %half_logdet : tensor<f64>
    return %neg : tensor<f64>
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

// CHECK-LABEL: func.func @main
// CHECK: %[[LD:.+]] = stablehlo.cholesky %{{.+}}, lower = true : tensor<3x3xf64>
// CHECK: enzymexla.lapack.syevd
// CHECK: impulse.infer_region @model
// CHECK-NOT: stablehlo.cholesky
// CHECK-NOT: enzymexla.lapack.potrf
// CHECK: } logpdf {
// CHECK-NOT: stablehlo.if
// CHECK-NOT: stablehlo.cholesky
// CHECK: impulse.yield

// -----

// K = s*C + t*D: fires with B = D.
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
    %B = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    %sb = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %sC = stablehlo.multiply %sb, %C : tensor<3x3xf64>
    %tb = stablehlo.broadcast_in_dim %t, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %tB = stablehlo.multiply %tb, %B : tensor<3x3xf64>
    %K = stablehlo.add %sC, %tB : tensor<3x3xf64>
    %L = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>
    %d = stablehlo.dot_general %L, %I, batching_dims = [1] x [1], contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3xf64>
    %logd = stablehlo.log %d : tensor<3xf64>
    %half_logdet = stablehlo.reduce(%logd init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %neg = stablehlo.negate %half_logdet : tensor<f64>
    return %neg : tensor<f64>
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

// CHECK-LABEL: func.func @main
// CHECK: %[[LD:.+]] = stablehlo.cholesky %{{.+}}, lower = true : tensor<3x3xf64>
// CHECK: enzymexla.lapack.syevd
// CHECK: impulse.infer_region @model
// CHECK-NOT: stablehlo.cholesky
// CHECK-NOT: enzymexla.lapack.potrf
// CHECK: } logpdf {
// CHECK-NOT: stablehlo.if
// CHECK-NOT: stablehlo.cholesky
// CHECK: impulse.yield

// -----

// K = t*W + s*C, W not positive definite: B = C, under the guard.
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
    %W = stablehlo.constant dense<[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]> : tensor<3x3xf64>
    %sb = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %sC = stablehlo.multiply %sb, %C : tensor<3x3xf64>
    %tb = stablehlo.broadcast_in_dim %t, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %tW = stablehlo.multiply %tb, %W : tensor<3x3xf64>
    %K = stablehlo.add %tW, %sC : tensor<3x3xf64>
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
    %y:2 = impulse.sample @y_sampler(%t#0, %s#1, %t#1, %C, %X) {
      logpdf = @y_logpdf,
      symbol = #impulse.symbol<3>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<3x3xf64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  func.func @main(%rng : tensor<2xui64>, %C : tensor<3x3xf64>, %X : tensor<3x3xf64>,
                  %trace : tensor<1x5xf64>)
      -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %result:4 = impulse.infer @model(%rng, %C, %X) given %trace
        init_step_size = %step_size {
      selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<3x3xf64>, tensor<3x3xf64>, tensor<1x5xf64>, tensor<f64>)
        -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %result#0, %result#1, %result#2, %result#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[C:[^:]+]]: tensor<3x3xf64>, %{{[^:]+}}: tensor<3x3xf64>, %{{[^:]+}}: tensor<1x5xf64>)
// CHECK: enzymexla.lapack.potrf %[[C]]
// CHECK-NOT: stablehlo.cholesky
// CHECK: enzymexla.lapack.syevd
// CHECK: } logpdf {
// CHECK: "stablehlo.if"
// CHECK: }, {
// CHECK: stablehlo.cholesky %{{.+}}, lower = true : tensor<3x3xf64>

// -----

// Must not fire: K = t*W + s*N, neither W nor N is positive definite.
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
    %W = stablehlo.constant dense<[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]> : tensor<3x3xf64>
    %N = stablehlo.constant dense<[[-1.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    %sb = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %sN = stablehlo.multiply %sb, %N : tensor<3x3xf64>
    %tb = stablehlo.broadcast_in_dim %t, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %tW = stablehlo.multiply %tb, %W : tensor<3x3xf64>
    %K = stablehlo.add %tW, %sN : tensor<3x3xf64>
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
    %y:2 = impulse.sample @y_sampler(%t#0, %s#1, %t#1, %C, %X) {
      logpdf = @y_logpdf,
      symbol = #impulse.symbol<3>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<3x3xf64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  func.func @main(%rng : tensor<2xui64>, %C : tensor<3x3xf64>, %X : tensor<3x3xf64>,
                  %trace : tensor<1x5xf64>)
      -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %result:4 = impulse.infer @model(%rng, %C, %X) given %trace
        init_step_size = %step_size {
      selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<3x3xf64>, tensor<3x3xf64>, tensor<1x5xf64>, tensor<f64>)
        -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %result#0, %result#1, %result#2, %result#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: enzymexla.lapack.syevd
// CHECK-NOT: enzymexla.lapack.potrf
// CHECK: } logpdf {
// CHECK-NOT: stablehlo.if
// CHECK: stablehlo.cholesky %{{.+}}, lower = true : tensor<3x3xf64>
// CHECK-NOT: enzymexla.lapack.syevd
