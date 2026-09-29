// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_triangular_solve_scale" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// triangular_solve(s * L, b) = (1/s) * triangular_solve(L, b), with L and b
// sample-invariant and s an effectively scalar sample-dependent quantity;
// no sign precondition (s != 0 is inherited from the original solve). A solve
// with unit_diagonal = true is left alone.
module {
  func.func private @halfnormal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %sample = stablehlo.abs %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }
  func.func private @halfnormal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  // CHECK-LABEL: func.func @scaled_coefficient_solve
  func.func @scaled_coefficient_solve(%rng : tensor<2xui64>, %L : tensor<3x3xf64>, %b : tensor<3x1xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    // CHECK: %[[ONE:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[L:.+]]: tensor<3x3xf64>, %[[B:.+]]: tensor<3x1xf64>):
    %res:8 = impulse.infer_region @model(%rng, %L, %b) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_L: tensor<3x3xf64>, %arg_b: tensor<3x1xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%lx: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %lx : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
      // CHECK: %[[TAUSQ:.+]] = stablehlo.multiply %[[TAU]]#1, %[[TAU]]#1 : tensor<f64>
      %tausq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
      %bcast = stablehlo.broadcast_in_dim %tausq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %scaledL = stablehlo.multiply %bcast, %arg_L : tensor<3x3xf64>
      // CHECK: %[[SOLVE:.+]] = "stablehlo.triangular_solve"(%[[L]], %[[B]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
      // CHECK-NEXT: %[[INV:.+]] = stablehlo.divide %[[ONE]], %[[TAUSQ]] : tensor<f64>
      // CHECK-NEXT: %[[INVB:.+]] = stablehlo.broadcast_in_dim %[[INV]], dims = [] : (tensor<f64>) -> tensor<3x1xf64>
      // CHECK-NEXT: %[[RES:.+]] = stablehlo.multiply %[[INVB]], %[[SOLVE]] : tensor<3x1xf64>
      %y = "stablehlo.triangular_solve"(%scaledL, %arg_b) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
      // CHECK: impulse.yield %[[TAU]]#0, %[[RES]] : tensor<2xui64>, tensor<3x1xf64>
      impulse.yield %tau#0, %y : tensor<2xui64>, tensor<3x1xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<3x3xf64>, tensor<3x1xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @unit_diagonal_solve_kept
  func.func @unit_diagonal_solve_kept(%rng : tensor<2xui64>, %L : tensor<3x3xf64>, %b : tensor<3x1xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    // Must not fire: unit_diagonal sees strict(s * L) + I, not a multiple of L.
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[L:.+]]: tensor<3x3xf64>, %[[B:.+]]: tensor<3x1xf64>):
    %res:8 = impulse.infer_region @model(%rng, %L, %b) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_L: tensor<3x3xf64>, %arg_b: tensor<3x1xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%lx: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %lx : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
      // CHECK: %[[TAUSQ:.+]] = stablehlo.multiply %[[TAU]]#1, %[[TAU]]#1 : tensor<f64>
      // CHECK-NEXT: %[[BC:.+]] = stablehlo.broadcast_in_dim %[[TAUSQ]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      // CHECK-NEXT: %[[SL:.+]] = stablehlo.multiply %[[BC]], %[[L]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[Y:.+]] = "stablehlo.triangular_solve"(%[[SL]], %[[B]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
      // CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[Y]] : tensor<2xui64>, tensor<3x1xf64>
      %tausq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
      %bcast = stablehlo.broadcast_in_dim %tausq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %scaledL = stablehlo.multiply %bcast, %arg_L : tensor<3x3xf64>
      %y = "stablehlo.triangular_solve"(%scaledL, %arg_b) {left_side = true, lower = true, unit_diagonal = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
      impulse.yield %tau#0, %y : tensor<2xui64>, tensor<3x1xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<3x3xf64>, tensor<3x1xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
