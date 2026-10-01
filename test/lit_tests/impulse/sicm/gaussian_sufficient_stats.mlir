// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_gaussian_sufficient_stats" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// ||c - R beta||^2 + rem: enzymexla.linalg.qr X is emitted (its Q1/R/c/rem are
// sample-invariant and hoist), and the in-loop chain is a D-vector
module {
  func.func private @normal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %std : tensor<2xui64>, tensor<f64>
  }
  func.func private @normal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  // CHECK-LABEL: func.func @linreg_qr
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[X:.+]]: tensor<64x3xf64>, %[[Y:.+]]: tensor<64xf64>):
  // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
  // The invariant thin-QR statistics (hoisted on the next fixpoint iteration):
  // R = Q1'X, c = Q1'y, and the stable remainder ||y - Q1 c||^2.
  // CHECK: %[[BETA:.+]] = stablehlo.broadcast_in_dim %[[TAU]]#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[Q:.+]], %[[RPACK:.+]], %[[INFO:.+]] = enzymexla.linalg.qr %[[X]] : (tensor<64x3xf64>) -> (tensor<64x3xf64>, tensor<64x3xf64>, tensor<i32>)
  // CHECK-NEXT: %[[RMAT:.+]] = stablehlo.dot_general %[[Q]], %[[X]], contracting_dims = [0] x [0] : (tensor<64x3xf64>, tensor<64x3xf64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[C:.+]] = stablehlo.dot_general %[[Q]], %[[Y]], contracting_dims = [0] x [0] : (tensor<64x3xf64>, tensor<64xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[Q1C:.+]] = stablehlo.dot_general %[[Q]], %[[C]], contracting_dims = [1] x [0] : (tensor<64x3xf64>, tensor<3xf64>) -> tensor<64xf64>
  // CHECK-NEXT: %[[RESID:.+]] = stablehlo.subtract %[[Y]], %[[Q1C]] : tensor<64xf64>
  // CHECK-NEXT: %[[REM:.+]] = stablehlo.dot_general %[[RESID]], %[[RESID]], contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[RB:.+]] = stablehlo.dot_general %[[RMAT]], %[[BETA]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[D:.+]] = stablehlo.subtract %[[C]], %[[RB]] : tensor<3xf64>
  // CHECK-NEXT: %[[DD:.+]] = stablehlo.dot_general %[[D]], %[[D]], contracting_dims = [0] x [0] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[RSS:.+]] = stablehlo.add %[[DD]], %[[REM]] : tensor<f64>
  // CHECK-NEXT: %[[SIG2:.+]] = stablehlo.multiply %[[TAU]]#1, %[[TAU]]#1 : tensor<f64>
  // CHECK-NEXT: %[[SUMSQ:.+]] = stablehlo.divide %[[RSS]], %[[SIG2]] : tensor<f64>
  // The original O(N D) mean contraction dot_general(X, beta) is gone.
  // CHECK-NOT: stablehlo.dot_general %[[X]], %[[BETA]]
  func.func @linreg_qr(%rng : tensor<2xui64>, %X : tensor<64x3xf64>, %y : tensor<64xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %X, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_X: tensor<64x3xf64>, %arg_y: tensor<64xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        impulse.yield %r, %s : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%lx: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %lx : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %beta = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %mu = stablehlo.dot_general %arg_X, %beta, contracting_dims = [1] x [0] : (tensor<64x3xf64>, tensor<3xf64>) -> tensor<64xf64>
      %cen = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %sigma_b = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<64xf64>
      %z = stablehlo.divide %cen, %sigma_b : tensor<64xf64>
      %z2 = stablehlo.multiply %z, %z : tensor<64xf64>
      %log2pi = stablehlo.constant dense<1.8378770664093453> : tensor<64xf64>
      %sm = stablehlo.add %z2, %log2pi : tensor<64xf64>
      %neghalf = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
      %neghalf_b = stablehlo.broadcast_in_dim %neghalf, dims = [] : (tensor<f64>) -> tensor<64xf64>
      %kernel = stablehlo.multiply %sm, %neghalf_b : tensor<64xf64>
      %ones = stablehlo.constant dense<1.000000e+00> : tensor<64xf64>
      %ss = stablehlo.dot_general %kernel, %ones, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<3xf64>
      impulse.yield %tau#0, %ss_b : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<64x3xf64>, tensor<64xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // The reduce{add} root from canonicalize; the outer -0.5 multiply stays.
  // CHECK-LABEL: func.func @linreg_qr_reduce
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[X:.+]]: tensor<64x3xf64>, %[[Y:.+]]: tensor<64xf64>):
  // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
  // CHECK: %[[Q:.+]], %{{.+}}, %{{.+}} = enzymexla.linalg.qr %[[X]] : (tensor<64x3xf64>) -> (tensor<64x3xf64>, tensor<64x3xf64>, tensor<i32>)
  // CHECK: %[[RB:.+]] = stablehlo.dot_general %{{.+}}, %{{.+}}, contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
  // CHECK-NOT: stablehlo.reduce
  // CHECK-NOT: stablehlo.dot_general %[[X]], %{{.+}} : (tensor<64x3xf64>, tensor<3xf64>)
  func.func @linreg_qr_reduce(%rng : tensor<2xui64>, %X : tensor<64x3xf64>, %y : tensor<64xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %X, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_X: tensor<64x3xf64>, %arg_y: tensor<64xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        impulse.yield %r, %s : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%lx: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %lx : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %beta = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %mu = stablehlo.dot_general %arg_X, %beta, contracting_dims = [1] x [0] : (tensor<64x3xf64>, tensor<3xf64>) -> tensor<64xf64>
      %cen = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %sigma_b = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<64xf64>
      %z = stablehlo.divide %cen, %sigma_b : tensor<64xf64>
      %z2 = stablehlo.multiply %z, %z : tensor<64xf64>
      %log2pi = stablehlo.constant dense<1.8378770664093453> : tensor<64xf64>
      %sm = stablehlo.add %z2, %log2pi : tensor<64xf64>
      %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %ss = stablehlo.reduce(%sm init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<64xf64>, tensor<f64>) -> tensor<f64>
      %neghalf = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
      %neg = stablehlo.multiply %ss, %neghalf : tensor<f64>
      %neg_b = stablehlo.broadcast_in_dim %neg, dims = [] : (tensor<f64>) -> tensor<3xf64>
      impulse.yield %tau#0, %neg_b : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<64x3xf64>, tensor<64xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @neg_x_dependent
  // CHECK-NOT: enzymexla.linalg.qr
  func.func @neg_x_dependent(%rng : tensor<2xui64>, %X : tensor<64x3xf64>, %y : tensor<64xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %X, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_X: tensor<64x3xf64>, %arg_y: tensor<64xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        impulse.yield %r, %s : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%lx: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %lx : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %beta = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %tau_mat = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<64x3xf64>
      %Xdep = stablehlo.multiply %tau_mat, %arg_X : tensor<64x3xf64>
      %mu = stablehlo.dot_general %Xdep, %beta, contracting_dims = [1] x [0] : (tensor<64x3xf64>, tensor<3xf64>) -> tensor<64xf64>
      %cen = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %z2 = stablehlo.multiply %cen, %cen : tensor<64xf64>
      %neghalf = stablehlo.constant dense<-5.000000e-01> : tensor<64xf64>
      %kernel = stablehlo.multiply %z2, %neghalf : tensor<64xf64>
      %ones = stablehlo.constant dense<1.000000e+00> : tensor<64xf64>
      %ss = stablehlo.dot_general %kernel, %ones, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<3xf64>
      impulse.yield %tau#0, %ss_b : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<64x3xf64>, tensor<64xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // Must not fire: N = 4 < 2 * D = 6.
  // CHECK-LABEL: func.func @neg_small_n
  // CHECK-NOT: enzymexla.linalg.qr
  func.func @neg_small_n(%rng : tensor<2xui64>, %X : tensor<4x3xf64>, %y : tensor<4xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %X, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_X: tensor<4x3xf64>, %arg_y: tensor<4xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        impulse.yield %r, %s : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%lx: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %lx : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %beta = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %mu = stablehlo.dot_general %arg_X, %beta, contracting_dims = [1] x [0] : (tensor<4x3xf64>, tensor<3xf64>) -> tensor<4xf64>
      %cen = stablehlo.subtract %arg_y, %mu : tensor<4xf64>
      %z2 = stablehlo.multiply %cen, %cen : tensor<4xf64>
      %neghalf = stablehlo.constant dense<-5.000000e-01> : tensor<4xf64>
      %kernel = stablehlo.multiply %z2, %neghalf : tensor<4xf64>
      %ones = stablehlo.constant dense<1.000000e+00> : tensor<4xf64>
      %ss = stablehlo.dot_general %kernel, %ones, contracting_dims = [0] x [0] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<3xf64>
      impulse.yield %tau#0, %ss_b : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<4x3xf64>, tensor<4xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @neg_no_square
  // CHECK-NOT: enzymexla.linalg.qr
  func.func @neg_no_square(%rng : tensor<2xui64>, %X : tensor<64x3xf64>, %y : tensor<64xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %X, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_X: tensor<64x3xf64>, %arg_y: tensor<64xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        impulse.yield %r, %s : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%lx: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %lx : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %beta = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %mu = stablehlo.dot_general %arg_X, %beta, contracting_dims = [1] x [0] : (tensor<64x3xf64>, tensor<3xf64>) -> tensor<64xf64>
      %cen = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %neghalf = stablehlo.constant dense<-5.000000e-01> : tensor<64xf64>
      %kernel = stablehlo.multiply %cen, %neghalf : tensor<64xf64>
      %ones = stablehlo.constant dense<1.000000e+00> : tensor<64xf64>
      %ss = stablehlo.dot_general %kernel, %ones, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<3xf64>
      impulse.yield %tau#0, %ss_b : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<64x3xf64>, tensor<64xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
