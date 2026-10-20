// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_cholesky_eigen_lift" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func private @halfnormal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %sample = stablehlo.abs %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }
  func.func private @halfnormal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  // CHECK-LABEL: func.func @eigen_lift_full_solve
  func.func @eigen_lift_full_solve(%rng : tensor<2xui64>, %A : tensor<3x3xf64>, %b : tensor<3x1xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    // CHECK: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[A:.+]]: tensor<3x3xf64>, %[[B:.+]]: tensor<3x1xf64>):
    %res:8 = impulse.infer_region @model(%rng, %A, %b) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<3x3xf64>, %arg_b: tensor<3x1xf64>):
      %I = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
      %t = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
      %s_bcast = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %sA = stablehlo.multiply %s_bcast, %arg_A : tensor<3x3xf64>
      %t_bcast = stablehlo.broadcast_in_dim %t, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %tI = stablehlo.multiply %t_bcast, %I : tensor<3x3xf64>
      %K = stablehlo.add %sA, %tI : tensor<3x3xf64>
      %F = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>
      // CHECK: %[[T:.+]] = stablehlo.multiply %[[TAU]]#1, %[[TAU]]#1 : tensor<f64>
      // CHECK-NEXT: %[[Q:[^,]+]], %[[LAMBDA:[^,]+]], %{{[^ ]+}} = enzymexla.lapack.syevd %[[A]] <uplo = L> : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xf64>, tensor<i64>)
      // CHECK-NEXT: %[[SB:.+]] = stablehlo.broadcast_in_dim %[[TAU]]#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
      // CHECK-NEXT: %[[SL:.+]] = stablehlo.multiply %[[SB]], %[[LAMBDA]] : tensor<3xf64>
      // CHECK-NEXT: %[[TB:.+]] = stablehlo.broadcast_in_dim %[[T]], dims = [] : (tensor<f64>) -> tensor<3xf64>
      // CHECK-NEXT: %[[D:.+]] = stablehlo.add %[[SL]], %[[TB]] : tensor<3xf64>
      // CHECK-NEXT: %[[QT:.+]] = stablehlo.transpose %[[Q]], dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
      // CHECK-NEXT: %[[INVD:.+]] = stablehlo.divide %[[ONES]], %[[D]] : tensor<3xf64>
      // CHECK-NEXT: %[[QTB:.+]] = stablehlo.dot_general %[[QT]], %[[B]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
      // CHECK-NEXT: %[[INVDB:.+]] = stablehlo.broadcast_in_dim %[[INVD]], dims = [0] : (tensor<3xf64>) -> tensor<3x1xf64>
      // CHECK-NEXT: %[[SCALED:.+]] = stablehlo.multiply %[[INVDB]], %[[QTB]] : tensor<3x1xf64>
      // CHECK-NEXT: %[[RES:.+]] = stablehlo.dot_general %[[Q]], %[[SCALED]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
      %inner = "stablehlo.triangular_solve"(%F, %arg_b) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
      %outer = "stablehlo.triangular_solve"(%F, %inner) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose ADJOINT>} : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
      // CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[RES]] : tensor<2xui64>, tensor<3x1xf64>
      impulse.yield %tau#0, %outer : tensor<2xui64>, tensor<3x1xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<3x3xf64>, tensor<3x1xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // For real factors TRANSPOSE equals ADJOINT, so the lift must still fire.
  // CHECK-LABEL: func.func @eigen_lift_full_solve_transpose
  func.func @eigen_lift_full_solve_transpose(%rng : tensor<2xui64>, %A : tensor<3x3xf64>, %b : tensor<3x1xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[A:.+]]: tensor<3x3xf64>, %[[B:.+]]: tensor<3x1xf64>):
    %res:8 = impulse.infer_region @model2(%rng, %A, %b) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<3x3xf64>, %arg_b: tensor<3x1xf64>):
      %I = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %t = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
      %s_bcast = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %sA = stablehlo.multiply %s_bcast, %arg_A : tensor<3x3xf64>
      %t_bcast = stablehlo.broadcast_in_dim %t, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %tI = stablehlo.multiply %t_bcast, %I : tensor<3x3xf64>
      %K = stablehlo.add %sA, %tI : tensor<3x3xf64>
      %F = stablehlo.cholesky %K, lower = false : tensor<3x3xf64>
      // CHECK: %[[Q:[^,]+]], %[[LAMBDA:[^,]+]], %{{[^ ]+}} = enzymexla.lapack.syevd %[[A]]
      // CHECK-NOT: stablehlo.cholesky
      // CHECK-NOT: stablehlo.triangular_solve
      %inner = "stablehlo.triangular_solve"(%F, %arg_b) {left_side = true, lower = false, unit_diagonal = false, transpose_a = #stablehlo<transpose TRANSPOSE>} : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
      %outer = "stablehlo.triangular_solve"(%F, %inner) {left_side = true, lower = false, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
      impulse.yield %tau#0, %outer : tensor<2xui64>, tensor<3x1xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<3x3xf64>, tensor<3x1xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
