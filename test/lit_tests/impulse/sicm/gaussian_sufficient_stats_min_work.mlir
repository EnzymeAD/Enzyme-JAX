// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_gaussian_sufficient_stats" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s --check-prefix=DEFAULT
// RUN: enzymexlamlir-opt %s --sicm="sufficient-stats-min-work=16" | FileCheck %s --check-prefix=LOW
// RUN: enzymexlamlir-opt %s --sicm="sufficient-stats-min-work=100" | FileCheck %s --check-prefix=HIGH

// Fires when N * D reaches min-work (default 64): @below has 22, @at has 64.

// DEFAULT-LABEL: func.func @below
// DEFAULT-NOT: enzymexla.linalg.qr
// DEFAULT: stablehlo.dot_general %[[R:.+]], %[[R]], contracting_dims = [0] x [0] : (tensor<11xf64>, tensor<11xf64>) -> tensor<f64>
// DEFAULT-LABEL: func.func @at
// DEFAULT: enzymexla.linalg.qr %{{.+}} : (tensor<32x3xf64>) -> (tensor<32x3xf64>, tensor<32x3xf64>, tensor<i32>)

// LOW-LABEL: func.func @below
// LOW: enzymexla.linalg.qr %{{.+}} : (tensor<11x3xf64>) -> (tensor<11x3xf64>, tensor<11x3xf64>, tensor<i32>)
// LOW-LABEL: func.func @at
// LOW: enzymexla.linalg.qr %{{.+}} : (tensor<32x3xf64>) -> (tensor<32x3xf64>, tensor<32x3xf64>, tensor<i32>)

// HIGH-LABEL: func.func @below
// HIGH-NOT: enzymexla.linalg.qr
// HIGH-LABEL: func.func @at
// HIGH-NOT: enzymexla.linalg.qr

module {
  func.func private @normal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %std : tensor<2xui64>, tensor<f64>
  }
  func.func private @normal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  func.func @below(%rng : tensor<2xui64>, %c1 : tensor<11xf64>, %y : tensor<11xf64>, %trace : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %c1, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_c1: tensor<11xf64>, %arg_y: tensor<11xf64>):
      %s1:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lx: tensor<1xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
      %s2:2 = impulse.sample_region(%s1#0, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lx: tensor<1xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<2>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
      %b1 = stablehlo.broadcast_in_dim %s1#1, dims = [0] : (tensor<1xf64>) -> tensor<11xf64>
      %m1 = stablehlo.multiply %b1, %arg_c1 : tensor<11xf64>
      %b2 = stablehlo.broadcast_in_dim %s2#1, dims = [0] : (tensor<1xf64>) -> tensor<11xf64>
      %mu = stablehlo.add %m1, %b2 : tensor<11xf64>
      %r = stablehlo.subtract %arg_y, %mu : tensor<11xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<11xf64>, tensor<11xf64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<2xf64>
      impulse.yield %s2#0, %ss_b : tensor<2xui64>, tensor<2xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]}> : (tensor<2xui64>, tensor<11xf64>, tensor<11xf64>, tensor<1x2xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @at(%rng : tensor<2xui64>, %c1 : tensor<32xf64>, %y : tensor<32xf64>, %trace : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %c1, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_c1: tensor<32xf64>, %arg_y: tensor<32xf64>):
      %s1:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lx: tensor<1xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
      %s2:2 = impulse.sample_region(%s1#0, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lx: tensor<1xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<2>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
      %b1 = stablehlo.broadcast_in_dim %s1#1, dims = [0] : (tensor<1xf64>) -> tensor<32xf64>
      %m1 = stablehlo.multiply %b1, %arg_c1 : tensor<32xf64>
      %b2 = stablehlo.broadcast_in_dim %s2#1, dims = [0] : (tensor<1xf64>) -> tensor<32xf64>
      %mu = stablehlo.add %m1, %b2 : tensor<32xf64>
      %r = stablehlo.subtract %arg_y, %mu : tensor<32xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<32xf64>, tensor<32xf64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<2xf64>
      impulse.yield %s2#0, %ss_b : tensor<2xui64>, tensor<2xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]}> : (tensor<2xui64>, tensor<32xf64>, tensor<32xf64>, tensor<1x2xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

}
