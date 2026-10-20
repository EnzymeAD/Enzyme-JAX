// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_sqrt_scale" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func private @halfnormal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %sample = stablehlo.abs %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }
  func.func private @halfnormal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  // CHECK-LABEL: func.func @sampler_sum_of_positive_draws
  func.func @sampler_sum_of_positive_draws(%rng : tensor<2xui64>, %R : tensor<3x3xf64>, %trace : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[R:.+]]: tensor<3x3xf64>):
    %res:8 = impulse.infer_region @model(%rng, %R) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_R: tensor<3x3xf64>):
      %tau1:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %tau2:2 = impulse.sample_region(%tau1#0, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<2>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[S:.+]] = stablehlo.add %{{.+}}#1, %{{.+}}#1 {{.*}}: tensor<f64>
      // CHECK: %[[SQRTA:.+]] = stablehlo.sqrt %[[R]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[SQRTS:.+]] = stablehlo.sqrt %[[S]] : tensor<f64>
      // CHECK-NEXT: %[[SQRTSB:.+]] = stablehlo.broadcast_in_dim %[[SQRTS]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      // CHECK-NEXT: %[[OUT:.+]] = stablehlo.multiply %[[SQRTSB]], %[[SQRTA]] : tensor<3x3xf64>
      %s = stablehlo.add %tau1#1, %tau2#1 : tensor<f64>
      %bcast = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %K = stablehlo.multiply %bcast, %arg_R : tensor<3x3xf64>
      %root = stablehlo.sqrt %K : tensor<3x3xf64>
      impulse.yield %tau2#0, %root : tensor<2xui64>, tensor<3x3xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]}> : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x2xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @logpdf_sum_of_positive_args
  // CHECK-SAME: (%{{.+}}: tensor<2xui64>, %[[LR:.+]]: tensor<3x3xf64>, %{{.+}}: tensor<1x2xf64>)
  func.func @logpdf_sum_of_positive_args(%rng : tensor<2xui64>, %R : tensor<3x3xf64>, %trace : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %R) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_R: tensor<3x3xf64>):
      %tau1:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } <{address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %tau2:2 = impulse.sample_region(%tau1#0, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } <{address = [#impulse.symbol<2>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %tau2#0, %tau2#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    // CHECK: impulse.yield %{{.+}}#0, %{{.+}}#1 : tensor<2xui64>, tensor<f64>
    // CHECK-NEXT: } logpdf {
    // CHECK-NEXT: ^bb0(%[[A1:.+]]: tensor<f64>, %[[A2:.+]]: tensor<f64>):
    ^bb0(%p1: tensor<f64>, %p2: tensor<f64>):
      // CHECK: %[[LS:.+]] = stablehlo.add %[[A1]], %[[A2]] {{.*}}: tensor<f64>
      // CHECK: %[[LSQRTA:.+]] = stablehlo.sqrt %[[LR]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[LSQRTS:.+]] = stablehlo.sqrt %[[LS]] : tensor<f64>
      // CHECK-NEXT: %[[LSQRTSB:.+]] = stablehlo.broadcast_in_dim %[[LSQRTS]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      // CHECK-NEXT: %[[LOUT:.+]] = stablehlo.multiply %[[LSQRTSB]], %[[LSQRTA]] : tensor<3x3xf64>
      %s = stablehlo.add %p1, %p2 : tensor<f64>
      %bcast = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %K = stablehlo.multiply %bcast, %R : tensor<3x3xf64>
      %root = stablehlo.sqrt %K : tensor<3x3xf64>
      %sum = stablehlo.reduce(%root init: %step_size) applies stablehlo.add across dimensions = [0, 1] : (tensor<3x3xf64>, tensor<f64>) -> tensor<f64>
      impulse.yield %sum : tensor<f64>
    } <{all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, num_position_args = 2 : i64, position_size = 2 : i64, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]}> : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x2xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
