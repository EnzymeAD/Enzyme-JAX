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
  func.func private @beta_sampler(%rng : tensor<2xui64>, %std : tensor<2xf64>)
      -> (tensor<2xui64>, tensor<2xf64>) {
    %sample = stablehlo.abs %std : tensor<2xf64>
    return %rng, %sample : tensor<2xui64>, tensor<2xf64>
  }
  func.func private @beta_logpdf(%x : tensor<2xf64>, %std : tensor<2xf64>)
      -> tensor<f64> {
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %sq = stablehlo.multiply %x, %x : tensor<2xf64>
    %sum = stablehlo.reduce(%sq init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
    return %sum : tensor<f64>
  }
  func.func private @beta3_sampler(%rng : tensor<2xui64>, %std : tensor<3xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %sample = stablehlo.abs %std : tensor<3xf64>
    return %rng, %sample : tensor<2xui64>, tensor<3xf64>
  }
  func.func private @beta3_logpdf(%x : tensor<3xf64>, %std : tensor<3xf64>)
      -> tensor<f64> {
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %sq = stablehlo.multiply %x, %x : tensor<3xf64>
    %sum = stablehlo.reduce(%sq init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    return %sum : tensor<f64>
  }
  func.func private @y_sampler(%rng : tensor<2xui64>, %s : tensor<f64>, %t : tensor<f64>,
                               %beta : tensor<2xf64>, %C : tensor<3x3xf64>, %X : tensor<3x2xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %zero = stablehlo.constant dense<0.0> : tensor<3xf64>
    return %rng, %zero : tensor<2xui64>, tensor<3xf64>
  }
  func.func private @y_logpdf(%y : tensor<3xf64>, %s : tensor<f64>, %t : tensor<f64>,
                              %beta : tensor<2xf64>, %C : tensor<3x3xf64>, %X : tensor<3x2xf64>)
      -> tensor<f64> {
    %I = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %sb = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %sC = stablehlo.multiply %sb, %C : tensor<3x3xf64>
    %tb = stablehlo.broadcast_in_dim %t, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %tI = stablehlo.multiply %tb, %I : tensor<3x3xf64>
    %K = stablehlo.add %sC, %tI : tensor<3x3xf64>
    %L = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>
    %mu = stablehlo.dot_general %X, %beta, contracting_dims = [1] x [0] : (tensor<3x2xf64>, tensor<2xf64>) -> tensor<3xf64>
    %res = stablehlo.subtract %y, %mu : tensor<3xf64>
    %rcol = stablehlo.reshape %res : (tensor<3xf64>) -> tensor<3x1xf64>
    %w = "stablehlo.triangular_solve"(%L, %rcol) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
    %z = "stablehlo.triangular_solve"(%L, %w) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
    %bz = stablehlo.multiply %rcol, %z : tensor<3x1xf64>
    %q = stablehlo.reduce(%bz init: %zero) applies stablehlo.add across dimensions = [0, 1] : (tensor<3x1xf64>, tensor<f64>) -> tensor<f64>
    %mhalf = stablehlo.constant dense<-0.5> : tensor<f64>
    %quad = stablehlo.multiply %mhalf, %q : tensor<f64>
    %d = stablehlo.dot_general %L, %I, batching_dims = [1] x [1], contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3xf64>
    %logd = stablehlo.log %d : tensor<3xf64>
    %half_logdet = stablehlo.reduce(%logd init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %lp = stablehlo.subtract %quad, %half_logdet : tensor<f64>
    return %lp : tensor<f64>
  }
  func.func private @y3_sampler(%rng : tensor<2xui64>, %s : tensor<f64>, %t : tensor<f64>,
                                %beta : tensor<3xf64>, %C : tensor<3x3xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %zero = stablehlo.constant dense<0.0> : tensor<3xf64>
    return %rng, %zero : tensor<2xui64>, tensor<3xf64>
  }
  func.func private @y3_logpdf(%y : tensor<3xf64>, %s : tensor<f64>, %t : tensor<f64>,
                               %beta : tensor<3xf64>, %C : tensor<3x3xf64>)
      -> tensor<f64> {
    %I = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    %sb = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %sC = stablehlo.multiply %sb, %C : tensor<3x3xf64>
    %tb = stablehlo.broadcast_in_dim %t, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %tI = stablehlo.multiply %tb, %I : tensor<3x3xf64>
    %K = stablehlo.add %sC, %tI : tensor<3x3xf64>
    %L = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %X = stablehlo.slice %C [0:3, 0:2] : (tensor<3x3xf64>) -> tensor<3x2xf64>
    %b2 = stablehlo.slice %beta [0:2] : (tensor<3xf64>) -> tensor<2xf64>
    %mu = stablehlo.dot_general %X, %b2, contracting_dims = [1] x [0] : (tensor<3x2xf64>, tensor<2xf64>) -> tensor<3xf64>
    %res = stablehlo.subtract %y, %mu : tensor<3xf64>
    %rcol = stablehlo.reshape %res : (tensor<3xf64>) -> tensor<3x1xf64>
    %w = "stablehlo.triangular_solve"(%L, %rcol) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
    %z = "stablehlo.triangular_solve"(%L, %w) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
    %bz = stablehlo.multiply %rcol, %z : tensor<3x1xf64>
    %q = stablehlo.reduce(%bz init: %zero) applies stablehlo.add across dimensions = [0, 1] : (tensor<3x1xf64>, tensor<f64>) -> tensor<f64>
    %zs = stablehlo.reduce(%z init: %zero) applies stablehlo.add across dimensions = [0, 1] : (tensor<3x1xf64>, tensor<f64>) -> tensor<f64>
    %lp = stablehlo.add %q, %zs : tensor<f64>
    return %lp : tensor<f64>
  }

  func.func @model(%rng : tensor<2xui64>, %C : tensor<3x3xf64>, %X : tensor<3x2xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %std2 = stablehlo.constant dense<1.0> : tensor<2xf64>
    %s:2 = impulse.sample @halfnormal_sampler(%rng, %std) <{
      logpdf = @halfnormal_logpdf, symbol = #impulse.symbol<1>, support = #impulse.support<POSITIVE>
    }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = impulse.sample @halfnormal_sampler(%s#0, %std) <{
      logpdf = @halfnormal_logpdf, symbol = #impulse.symbol<2>, support = #impulse.support<POSITIVE>
    }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %beta:2 = impulse.sample @beta_sampler(%t#0, %std2) <{
      logpdf = @beta_logpdf, symbol = #impulse.symbol<3>, support = #impulse.support<REAL>
    }> : (tensor<2xui64>, tensor<2xf64>) -> (tensor<2xui64>, tensor<2xf64>)
    %y:2 = impulse.sample @y_sampler(%beta#0, %s#1, %t#1, %beta#1, %C, %X) <{
      logpdf = @y_logpdf, symbol = #impulse.symbol<4>
    }> : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<2xf64>, tensor<3x3xf64>, tensor<3x2xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  func.func @model_reused(%rng : tensor<2xui64>, %C : tensor<3x3xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %std3 = stablehlo.constant dense<1.0> : tensor<3xf64>
    %s:2 = impulse.sample @halfnormal_sampler(%rng, %std) <{
      logpdf = @halfnormal_logpdf, symbol = #impulse.symbol<1>, support = #impulse.support<POSITIVE>
    }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = impulse.sample @halfnormal_sampler(%s#0, %std) <{
      logpdf = @halfnormal_logpdf, symbol = #impulse.symbol<2>, support = #impulse.support<POSITIVE>
    }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %beta:2 = impulse.sample @beta3_sampler(%t#0, %std3) <{
      logpdf = @beta3_logpdf, symbol = #impulse.symbol<3>, support = #impulse.support<REAL>
    }> : (tensor<2xui64>, tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    %y:2 = impulse.sample @y3_sampler(%beta#0, %s#1, %t#1, %beta#1, %C) <{
      logpdf = @y3_logpdf, symbol = #impulse.symbol<4>
    }> : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<3xf64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  // CHECK-LABEL: func.func @main
  // CHECK-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[C:[^:]+]]: tensor<3x3xf64>, %[[X:[^:]+]]: tensor<3x2xf64>, %{{[^:]+}}: tensor<1x7xf64>)
  // CHECK: %[[YROW:.+]] = impulse.slice
  // CHECK-NEXT: %[[Y:.+]] = impulse.reshape %[[YROW]] : (tensor<1x3xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[Q:[^,]+]], %[[LAMBDA:[^,]+]], %{{[^ ]+}} = enzymexla.lapack.syevd %[[C]] <uplo = L> : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xf64>, tensor<i64>)
  // CHECK-NEXT: %[[QT:.+]] = stablehlo.transpose %[[Q]], dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[QTY:.+]] = stablehlo.dot_general %[[QT]], %[[Y]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[QTYCOL:.+]] = stablehlo.reshape %[[QTY]] : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[QTX:.+]] = stablehlo.dot_general %[[QT]], %[[X]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
  // CHECK-NEXT: impulse.infer_region @model
  // CHECK-NOT: stablehlo.cholesky
  // CHECK: impulse.yield %{{.+}}#0, %{{.+}}#1 : tensor<2xui64>, tensor<3xf64>
  // CHECK-NEXT: } logpdf {
  // CHECK-NEXT: ^bb0(%[[S:.+]]: tensor<f64>, %[[T:.+]]: tensor<f64>, %[[BETA:.+]]: tensor<2xf64>):
  // CHECK: %[[D:.+]] = stablehlo.add %{{.+}}, %{{.+}} : tensor<3xf64>
  // CHECK-NEXT: %[[SQD:.+]] = stablehlo.sqrt %[[D]] : tensor<3xf64>
  // CHECK-NEXT: %[[ISQD:.+]] = stablehlo.divide %{{.+}}, %[[SQD]] : tensor<3xf64>
  // CHECK-NEXT: %[[SQDD:.+]] = stablehlo.sqrt %[[D]] : tensor<3xf64>
  // CHECK-NEXT: %[[QTXB:.+]] = stablehlo.dot_general %[[QTX]], %[[BETA]], contracting_dims = [1] x [0] : (tensor<3x2xf64>, tensor<2xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[QTXBCOL:.+]] = stablehlo.reshape %[[QTXB]] : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[QTB:.+]] = stablehlo.subtract %[[QTYCOL]], %[[QTXBCOL]] : tensor<3x1xf64>
  // CHECK-NEXT: %[[ISQDB:.+]] = stablehlo.broadcast_in_dim %[[ISQD]], dims = [0] : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[U:.+]] = stablehlo.multiply %[[ISQDB]], %[[QTB]] : tensor<3x1xf64>
  // CHECK-NEXT: %[[QUAD:.+]] = stablehlo.dot_general %[[U]], %[[U]], contracting_dims = [0, 1] x [0, 1] : (tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<f64>
  // CHECK-NEXT: %{{.+}} = stablehlo.multiply %{{.+}}, %[[QUAD]] : tensor<f64>
  // CHECK-NEXT: %{{.+}} = stablehlo.log %[[SQDD]] : tensor<3xf64>
  func.func @main(%rng : tensor<2xui64>, %C : tensor<3x3xf64>, %X : tensor<3x2xf64>,
                  %trace : tensor<1x7xf64>)
      -> (tensor<1x4xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %result:4 = impulse.infer @model(%rng, %C, %X) given %trace
        init_step_size = %step_size <{
      selection = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64, thinning = 1 : i64, num_warmup = 0 : i64
    }> : (tensor<2xui64>, tensor<3x3xf64>, tensor<3x2xf64>, tensor<1x7xf64>, tensor<f64>)
        -> (tensor<1x4xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %result#0, %result#1, %result#2, %result#3 : tensor<1x4xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @main_reused
  // CHECK-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[C:[^:]+]]: tensor<3x3xf64>, %{{[^:]+}}: tensor<1x8xf64>)
  // CHECK: %[[Q:[^,]+]], %[[LAMBDA:[^,]+]], %{{[^ ]+}} = enzymexla.lapack.syevd %[[C]]
  // CHECK-NEXT: %[[QT:.+]] = stablehlo.transpose %[[Q]], dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[X:.+]] = stablehlo.slice %[[C]] [0:3, 0:2] : (tensor<3x3xf64>) -> tensor<3x2xf64>
  // CHECK-NEXT: %[[QTY:.+]] = stablehlo.dot_general %[[QT]], %{{.+}}, contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[QTYCOL:.+]] = stablehlo.reshape %[[QTY]] : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[QTX:.+]] = stablehlo.dot_general %[[QT]], %[[X]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
  // CHECK-NEXT: impulse.infer_region @model_reused
  // CHECK-NOT: stablehlo.cholesky
  // CHECK: impulse.yield %{{.+}}#0, %{{.+}}#1 : tensor<2xui64>, tensor<3xf64>
  // CHECK-NEXT: } logpdf {
  // CHECK-NEXT: ^bb0(%[[S:.+]]: tensor<f64>, %[[T:.+]]: tensor<f64>, %[[BETA:.+]]: tensor<3xf64>):
  // CHECK: %[[D:.+]] = stablehlo.add %{{.+}}, %{{.+}} : tensor<3xf64>
  // CHECK-NEXT: %[[INVD:.+]] = stablehlo.divide %{{.+}}, %[[D]] : tensor<3xf64>
  // CHECK-NEXT: %[[B2:.+]] = stablehlo.slice %[[BETA]] [0:2] : (tensor<3xf64>) -> tensor<2xf64>
  // CHECK-NEXT: %[[MU:.+]] = stablehlo.dot_general %[[X]], %[[B2]], contracting_dims = [1] x [0] : (tensor<3x2xf64>, tensor<2xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[RES:.+]] = stablehlo.subtract %{{.+}}, %[[MU]] : tensor<3xf64>
  // CHECK-NEXT: %[[RCOL:.+]] = stablehlo.reshape %[[RES]] : (tensor<3xf64>) -> tensor<3x1xf64>
  // The chain also feeds sum(z), so it is replaced whole: Q diag(1/d) Q^T b.
  // CHECK-NEXT: %[[QTXB:.+]] = stablehlo.dot_general %[[QTX]], %[[B2]], contracting_dims = [1] x [0] : (tensor<3x2xf64>, tensor<2xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[QTXBCOL:.+]] = stablehlo.reshape %[[QTXB]] : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[QTB:.+]] = stablehlo.subtract %[[QTYCOL]], %[[QTXBCOL]] : tensor<3x1xf64>
  // CHECK-NEXT: %[[INVDB:.+]] = stablehlo.broadcast_in_dim %[[INVD]], dims = [0] : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[SC:.+]] = stablehlo.multiply %[[INVDB]], %[[QTB]] : tensor<3x1xf64>
  // CHECK-NEXT: %[[Z:.+]] = stablehlo.dot_general %[[Q]], %[[SC]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[BZ:.+]] = stablehlo.multiply %[[RCOL]], %[[Z]] : tensor<3x1xf64>
  // CHECK-NEXT: %{{.+}} = stablehlo.reduce(%[[BZ]] init: %{{.+}}) applies stablehlo.add across dimensions = [0, 1] : (tensor<3x1xf64>, tensor<f64>) -> tensor<f64>
  // CHECK-NEXT: %{{.+}} = stablehlo.reduce(%[[Z]] init: %{{.+}}) applies stablehlo.add across dimensions = [0, 1] : (tensor<3x1xf64>, tensor<f64>) -> tensor<f64>
  func.func @main_reused(%rng : tensor<2xui64>, %C : tensor<3x3xf64>,
                         %trace : tensor<1x8xf64>)
      -> (tensor<1x5xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %result:4 = impulse.infer @model_reused(%rng, %C) given %trace
        init_step_size = %step_size <{
      selection = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64, thinning = 1 : i64, num_warmup = 0 : i64
    }> : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x8xf64>, tensor<f64>)
        -> (tensor<1x5xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %result#0, %result#1, %result#2, %result#3 : tensor<1x5xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}
