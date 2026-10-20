// RUN: enzymexlamlir-opt %s --split-input-file --inline-infer-regions --sicm="guard-versioning=false" | FileCheck %s

// K = s*C + t*D, D a model input: guarded by potrf(D) info == 0.
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

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[C:[^:]+]]: tensor<3x3xf64>, %[[D:[^:]+]]: tensor<3x3xf64>, %{{[^:]+}}: tensor<1x5xf64>)
// CHECK: %[[LB:.+]], %[[INFO:.+]] = enzymexla.lapack.potrf %[[D]] <uplo = L>
// CHECK-NEXT: %[[OK:.+]] = stablehlo.compare EQ, %[[INFO]], %{{.+}} : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK: %[[F:.+]] = stablehlo.select %[[OK]], %[[LB]], %{{.+}} : tensor<i1>, tensor<3x3xf64>
// CHECK-NEXT: %[[Y:.+]] = "stablehlo.triangular_solve"(%[[F]], %[[C]])
// CHECK-NEXT: %[[CW:.+]] = "stablehlo.triangular_solve"(%[[F]], %[[Y]])
// CHECK-NEXT: %{{.+}}, %[[LAM:.+]], %{{.+}} = enzymexla.lapack.syevd %[[CW]]
// CHECK-NEXT: %[[DLB:.+]] = stablehlo.dot_general %[[F]], %{{.+}}, batching_dims = [1] x [1], contracting_dims = [0] x [0]
// CHECK: impulse.yield %{{.+}}#0, %{{.+}}#1 : tensor<2xui64>, tensor<3xf64>
// CHECK-NEXT: } logpdf {
// CHECK-NEXT: ^bb0(%[[S:.+]]: tensor<f64>, %[[T:.+]]: tensor<f64>):
// CHECK-NEXT: %[[R:.+]] = "stablehlo.if"(%[[OK]]) ({
// CHECK-NEXT: %[[SB:.+]] = stablehlo.broadcast_in_dim %[[S]], dims = [] : (tensor<f64>) -> tensor<3xf64>
// CHECK-NEXT: %[[SL:.+]] = stablehlo.multiply %[[SB]], %[[LAM]] : tensor<3xf64>
// CHECK-NEXT: %[[TB:.+]] = stablehlo.broadcast_in_dim %[[T]], dims = [] : (tensor<f64>) -> tensor<3xf64>
// CHECK-NEXT: %[[SH:.+]] = stablehlo.add %[[SL]], %[[TB]] : tensor<3xf64>
// CHECK-NEXT: %[[SQ:.+]] = stablehlo.sqrt %[[SH]] : tensor<3xf64>
// CHECK-NEXT: %[[LIFT:.+]] = stablehlo.multiply %[[DLB]], %[[SQ]] : tensor<3xf64>
// CHECK-NEXT: stablehlo.return %[[LIFT]] : tensor<3xf64>
// CHECK-NEXT: }, {
// CHECK-NEXT: %[[SB2:.+]] = stablehlo.broadcast_in_dim %[[S]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
// CHECK-NEXT: %[[SC:.+]] = stablehlo.multiply %[[SB2]], %[[C]] : tensor<3x3xf64>
// CHECK-NEXT: %[[TB2:.+]] = stablehlo.broadcast_in_dim %[[T]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
// CHECK-NEXT: %[[TD:.+]] = stablehlo.multiply %[[TB2]], %[[D]] : tensor<3x3xf64>
// CHECK-NEXT: %[[K:.+]] = stablehlo.add %[[SC]], %[[TD]] : tensor<3x3xf64>
// CHECK-NEXT: %[[L:.+]] = stablehlo.cholesky %[[K]], lower = true : tensor<3x3xf64>
// CHECK-NEXT: %[[DL:.+]] = stablehlo.dot_general %[[L]], %{{.+}}, batching_dims = [1] x [1], contracting_dims = [0] x [0]
// CHECK-NEXT: stablehlo.return %[[DL]] : tensor<3xf64>
// CHECK-NEXT: }) {sicm.runtime_guard} : (tensor<i1>) -> tensor<3xf64>
// CHECK: %[[LOG:.+]] = stablehlo.log %[[R]] : tensor<3xf64>
// CHECK-NOT: sicm.eigenlift_guard

// -----

// K = t*D + s*C: B = C, the second summand's matrix, under the guard.
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
    %K = stablehlo.add %tX, %sC : tensor<3x3xf64>
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

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[C:[^:]+]]: tensor<3x3xf64>, %[[D:[^:]+]]: tensor<3x3xf64>, %{{[^:]+}}: tensor<1x5xf64>)
// CHECK: %[[LB:.+]], %[[INFO:.+]] = enzymexla.lapack.potrf %[[C]] <uplo = L>
// CHECK-NEXT: %[[OK:.+]] = stablehlo.compare EQ, %[[INFO]], %{{.+}} : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK: %[[F:.+]] = stablehlo.select %[[OK]], %[[LB]], %{{.+}} : tensor<i1>, tensor<3x3xf64>
// CHECK-NEXT: %[[Y:.+]] = "stablehlo.triangular_solve"(%[[F]], %[[D]])
// CHECK: impulse.yield %{{.+}}#0, %{{.+}}#1 : tensor<2xui64>, tensor<3xf64>
// CHECK-NEXT: } logpdf {
// CHECK-NEXT: ^bb0(%[[S:.+]]: tensor<f64>, %[[T:.+]]: tensor<f64>):
// CHECK-NEXT: %{{.+}} = "stablehlo.if"(%[[OK]]) ({
// CHECK: }, {
// CHECK: %[[K:.+]] = stablehlo.add %{{.+}}, %{{.+}} : tensor<3x3xf64>
// CHECK-NEXT: %{{.+}} = stablehlo.cholesky %[[K]], lower = true : tensor<3x3xf64>

// -----

// K = s*C + diag(t*v): B = diag(v) checked by min(v) > 0, no potrf.
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
                               %t : tensor<f64>, %C : tensor<3x3xf64>, %X : tensor<3xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %zero = stablehlo.constant dense<0.0> : tensor<3xf64>
    return %rng, %zero : tensor<2xui64>, tensor<3xf64>
  }

  func.func private @y_logpdf(%y : tensor<3xf64>, %s : tensor<f64>,
                              %t : tensor<f64>, %C : tensor<3x3xf64>, %X : tensor<3xf64>)
      -> tensor<f64> {
    %I = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %sb = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %sC = stablehlo.multiply %sb, %C : tensor<3x3xf64>
    %idx = stablehlo.iota dim = 0 : tensor<3x2xi64>
    %tv = stablehlo.broadcast_in_dim %t, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %u = stablehlo.multiply %tv, %X : tensor<3xf64>
    %K = "stablehlo.scatter"(%sC, %idx, %u) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%a : tensor<f64>, %b : tensor<f64>):
      %r = stablehlo.add %a, %b : tensor<f64>
      stablehlo.return %r : tensor<f64>
    }) : (tensor<3x3xf64>, tensor<3x2xi64>, tensor<3xf64>) -> tensor<3x3xf64>
    %L = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>
    %d = stablehlo.dot_general %L, %I, batching_dims = [1] x [1], contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3xf64>
    %logd = stablehlo.log %d : tensor<3xf64>
    %half_logdet = stablehlo.reduce(%logd init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %neg = stablehlo.negate %half_logdet : tensor<f64>
    return %neg : tensor<f64>
  }

  func.func @model(%rng : tensor<2xui64>, %C : tensor<3x3xf64>, %X : tensor<3xf64>)
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
    }> : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<3x3xf64>, tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  func.func @main(%rng : tensor<2xui64>, %C : tensor<3x3xf64>, %X : tensor<3xf64>,
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
    }> : (tensor<2xui64>, tensor<3x3xf64>, tensor<3xf64>, tensor<1x5xf64>, tensor<f64>)
        -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %result#0, %result#1, %result#2, %result#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[C:[^:]+]]: tensor<3x3xf64>, %[[V:[^:]+]]: tensor<3xf64>, %{{[^:]+}}: tensor<1x5xf64>)
// CHECK-NOT: enzymexla.lapack.potrf
// CHECK-NOT: stablehlo.triangular_solve
// CHECK: %[[MIN:.+]] = stablehlo.reduce(%[[V]] init: %{{.+}}) applies stablehlo.minimum across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[OK:.+]] = stablehlo.compare GT, %[[MIN]], %{{.+}} : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CHECK-NEXT: %[[SAFE:.+]] = stablehlo.select %[[OK]], %[[V]], %{{.+}} : tensor<i1>, tensor<3xf64>
// CHECK-NEXT: %[[R:.+]] = stablehlo.sqrt %[[SAFE]] : tensor<3xf64>
// CHECK-NEXT: %[[RR:.+]] = stablehlo.broadcast_in_dim %[[R]], dims = [0] : (tensor<3xf64>) -> tensor<3x3xf64>
// CHECK-NEXT: %[[RC:.+]] = stablehlo.broadcast_in_dim %[[R]], dims = [1] : (tensor<3xf64>) -> tensor<3x3xf64>
// CHECK-NEXT: %[[OUT:.+]] = stablehlo.multiply %[[RR]], %[[RC]] : tensor<3x3xf64>
// CHECK-NEXT: %[[CW:.+]] = stablehlo.divide %[[C]], %[[OUT]] : tensor<3x3xf64>
// CHECK-NEXT: %{{.+}}, %[[LAM:.+]], %{{.+}} = enzymexla.lapack.syevd %[[CW]]
// CHECK: } logpdf {
// CHECK: %{{.+}} = "stablehlo.if"(%[[OK]]) ({
// CHECK: %[[SQ:.+]] = stablehlo.sqrt %{{.+}} : tensor<3xf64>
// CHECK-NEXT: %[[LIFT:.+]] = stablehlo.multiply %[[R]], %[[SQ]] : tensor<3xf64>
// CHECK-NEXT: stablehlo.return %[[LIFT]] : tensor<3xf64>
// CHECK-NEXT: }, {
// CHECK: "stablehlo.scatter"
// CHECK: stablehlo.cholesky

// -----

// Constant positive v: B is proven positive definite, so no check or if.
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
                               %t : tensor<f64>, %C : tensor<3x3xf64>, %X : tensor<3xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %zero = stablehlo.constant dense<0.0> : tensor<3xf64>
    return %rng, %zero : tensor<2xui64>, tensor<3xf64>
  }

  func.func private @y_logpdf(%y : tensor<3xf64>, %s : tensor<f64>,
                              %t : tensor<f64>, %C : tensor<3x3xf64>, %X : tensor<3xf64>)
      -> tensor<f64> {
    %I = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %v = stablehlo.constant dense<[2.0, 3.0, 1.0]> : tensor<3xf64>
    %sb = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %sC = stablehlo.multiply %sb, %C : tensor<3x3xf64>
    %idx = stablehlo.iota dim = 0 : tensor<3x2xi64>
    %tv = stablehlo.broadcast_in_dim %t, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %u = stablehlo.multiply %tv, %v : tensor<3xf64>
    %K = "stablehlo.scatter"(%sC, %idx, %u) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%a : tensor<f64>, %b : tensor<f64>):
      %r = stablehlo.add %a, %b : tensor<f64>
      stablehlo.return %r : tensor<f64>
    }) : (tensor<3x3xf64>, tensor<3x2xi64>, tensor<3xf64>) -> tensor<3x3xf64>
    %L = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>
    %d = stablehlo.dot_general %L, %I, batching_dims = [1] x [1], contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3xf64>
    %logd = stablehlo.log %d : tensor<3xf64>
    %half_logdet = stablehlo.reduce(%logd init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %neg = stablehlo.negate %half_logdet : tensor<f64>
    return %neg : tensor<f64>
  }

  func.func @model(%rng : tensor<2xui64>, %C : tensor<3x3xf64>, %X : tensor<3xf64>)
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
    }> : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<3x3xf64>, tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  func.func @main(%rng : tensor<2xui64>, %C : tensor<3x3xf64>, %X : tensor<3xf64>,
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
    }> : (tensor<2xui64>, tensor<3x3xf64>, tensor<3xf64>, tensor<1x5xf64>, tensor<f64>)
        -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %result#0, %result#1, %result#2, %result#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: stablehlo.minimum
// CHECK: %[[V:.+]] = stablehlo.constant dense<[2.000000e+00, 3.000000e+00, 1.000000e+00]> : tensor<3xf64>
// CHECK: %[[R:.+]] = stablehlo.sqrt %[[V]] : tensor<3xf64>
// CHECK: enzymexla.lapack.syevd
// CHECK: } logpdf {
// CHECK-NOT: stablehlo.if
// CHECK-NOT: stablehlo.cholesky
// CHECK: %[[SQ:.+]] = stablehlo.sqrt %{{.+}} : tensor<3xf64>
// CHECK-NEXT: %{{.+}} = stablehlo.multiply %[[R]], %[[SQ]] : tensor<3xf64>
// CHECK: impulse.yield
