// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_gaussian_sufficient_stats" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// offset), so that c = Q1'y and the remainder stay functions of X and y only
module {
  func.func private @normal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %std : tensor<2xui64>, tensor<f64>
  }
  func.func private @normal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  // b is not loop-invariant: a beta entry on a ones column, not a y offset.
  // CHECK-LABEL: func.func @single_address
  // CHECK-DAG: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<64x1xf64>
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[C1:.+]]: tensor<64xf64>, %[[Y:.+]]: tensor<64xf64>, %[[B:.+]]: tensor<1xf64>):
  // CHECK: %[[S1:.+]]:2 = impulse.sample_region
  // CHECK: %[[C1COL:.+]] = stablehlo.reshape %[[C1]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[X:.+]] = stablehlo.concatenate %[[C1COL]], %[[ONES]], dim = 1 : (tensor<64x1xf64>, tensor<64x1xf64>) -> tensor<64x2xf64>
  // CHECK-NEXT: %[[BETA:.+]] = stablehlo.concatenate %[[S1]]#1, %[[B]], dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  // CHECK-NEXT: %[[Q:.+]], %{{.+}}, %{{.+}} = enzymexla.linalg.qr %[[X]] : (tensor<64x2xf64>) -> (tensor<64x2xf64>, tensor<64x2xf64>, tensor<i32>)
  // CHECK-NEXT: %{{.+}} = stablehlo.dot_general %[[Q]], %[[X]], contracting_dims = [0] x [0] : (tensor<64x2xf64>, tensor<64x2xf64>) -> tensor<2x2xf64>
  // CHECK-NEXT: %[[C:.+]] = stablehlo.dot_general %[[Q]], %[[Y]], contracting_dims = [0] x [0] : (tensor<64x2xf64>, tensor<64xf64>) -> tensor<2xf64>
  // CHECK-NEXT: %[[Q1C:.+]] = stablehlo.dot_general %[[Q]], %[[C]], contracting_dims = [1] x [0] : (tensor<64x2xf64>, tensor<2xf64>) -> tensor<64xf64>
  // CHECK-NEXT: %{{.+}} = stablehlo.subtract %[[Y]], %[[Q1C]] : tensor<64xf64>
  // CHECK: %{{.+}} = stablehlo.dot_general %{{.+}}, %[[BETA]], contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
  func.func @single_address(%rng : tensor<2xui64>, %c1 : tensor<64xf64>, %y : tensor<64xf64>, %b : tensor<1xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %c1, %y, %b) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_c1: tensor<64xf64>, %arg_y: tensor<64xf64>, %arg_b: tensor<1xf64>):
      %s1:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lx: tensor<1xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
      %b1 = stablehlo.broadcast_in_dim %s1#1, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %m1 = stablehlo.multiply %b1, %arg_c1 : tensor<64xf64>
      %bb = stablehlo.broadcast_in_dim %arg_b, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %mu = stablehlo.add %m1, %bb : tensor<64xf64>
      %r = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<1xf64>
      impulse.yield %s1#0, %ss_b : tensor<2xui64>, tensor<1xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<64xf64>, tensor<64xf64>, tensor<1xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @merged_ones
  // CHECK-DAG: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<64x1xf64>
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[C1:.+]]: tensor<64xf64>, %[[Y:.+]]: tensor<64xf64>, %[[B:.+]]: tensor<1xf64>):
  // CHECK: %[[S1:.+]]:2 = impulse.sample_region
  // CHECK: %[[S2:.+]]:2 = impulse.sample_region
  // CHECK: %[[C1COL:.+]] = stablehlo.reshape %[[C1]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[SUM:.+]] = stablehlo.subtract %[[S2]]#1, %[[B]] : tensor<1xf64>
  // CHECK-NEXT: %[[X:.+]] = stablehlo.concatenate %[[C1COL]], %[[ONES]], dim = 1 : (tensor<64x1xf64>, tensor<64x1xf64>) -> tensor<64x2xf64>
  // CHECK-NEXT: %[[BETA:.+]] = stablehlo.concatenate %[[S1]]#1, %[[SUM]], dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  // CHECK-NEXT: %{{.+}}, %{{.+}}, %{{.+}} = enzymexla.linalg.qr %[[X]] : (tensor<64x2xf64>) -> (tensor<64x2xf64>, tensor<64x2xf64>, tensor<i32>)
  func.func @merged_ones(%rng : tensor<2xui64>, %c1 : tensor<64xf64>, %y : tensor<64xf64>, %b : tensor<1xf64>, %trace : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %c1, %y, %b) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_c1: tensor<64xf64>, %arg_y: tensor<64xf64>, %arg_b: tensor<1xf64>):
      %s1:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lx: tensor<1xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
      %s2:2 = impulse.sample_region(%s1#0, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lx: tensor<1xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<2>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
      %b1 = stablehlo.broadcast_in_dim %s1#1, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %m1 = stablehlo.multiply %b1, %arg_c1 : tensor<64xf64>
      %b2 = stablehlo.broadcast_in_dim %s2#1, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %bb = stablehlo.broadcast_in_dim %arg_b, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %a1 = stablehlo.add %m1, %b2 : tensor<64xf64>
      %mu = stablehlo.subtract %a1, %bb : tensor<64xf64>
      %r = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<2xf64>
      impulse.yield %s2#0, %ss_b : tensor<2xui64>, tensor<2xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]} : (tensor<2xui64>, tensor<64xf64>, tensor<64xf64>, tensor<1xf64>, tensor<1x2xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @constant_offset
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[C1:.+]]: tensor<64xf64>, %[[Y:.+]]: tensor<64xf64>, %{{.+}}: tensor<1xf64>):
  // CHECK: %[[S1:.+]]:2 = impulse.sample_region
  // CHECK: %[[K:.+]] = stablehlo.broadcast_in_dim %{{.+}}, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
  // CHECK-NEXT: %[[X:.+]] = stablehlo.reshape %[[C1]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[YP:.+]] = stablehlo.subtract %[[Y]], %[[K]] : tensor<64xf64>
  // CHECK-NEXT: %[[Q:.+]], %{{.+}}, %{{.+}} = enzymexla.linalg.qr %[[X]] : (tensor<64x1xf64>) -> (tensor<64x1xf64>, tensor<64x1xf64>, tensor<i32>)
  // CHECK-NEXT: %{{.+}} = stablehlo.dot_general %[[Q]], %[[X]], contracting_dims = [0] x [0] : (tensor<64x1xf64>, tensor<64x1xf64>) -> tensor<1x1xf64>
  // CHECK-NEXT: %{{.+}} = stablehlo.dot_general %[[Q]], %[[YP]], contracting_dims = [0] x [0] : (tensor<64x1xf64>, tensor<64xf64>) -> tensor<1xf64>
  func.func @constant_offset(%rng : tensor<2xui64>, %c1 : tensor<64xf64>, %y : tensor<64xf64>, %b : tensor<1xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %c1, %y, %b) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_c1: tensor<64xf64>, %arg_y: tensor<64xf64>, %arg_b: tensor<1xf64>):
      %s1:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lx: tensor<1xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
      %b1 = stablehlo.broadcast_in_dim %s1#1, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %m1 = stablehlo.multiply %b1, %arg_c1 : tensor<64xf64>
      %k = stablehlo.constant dense<3.0> : tensor<1xf64>
      %bk = stablehlo.broadcast_in_dim %k, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %mu = stablehlo.add %m1, %bk : tensor<64xf64>
      %r = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<1xf64>
      impulse.yield %s1#0, %ss_b : tensor<2xui64>, tensor<1xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<64xf64>, tensor<64xf64>, tensor<1xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @neg_no_dependent
  // CHECK-NOT: enzymexla.linalg.qr
  // CHECK: stablehlo.dot_general %[[R:.+]], %[[R]], contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
  func.func @neg_no_dependent(%rng : tensor<2xui64>, %c1 : tensor<64xf64>, %y : tensor<64xf64>, %b : tensor<1xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %c1, %y, %b) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_c1: tensor<64xf64>, %arg_y: tensor<64xf64>, %arg_b: tensor<1xf64>):
      %s1:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lx: tensor<1xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
      %bb = stablehlo.broadcast_in_dim %arg_b, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %m1 = stablehlo.multiply %bb, %arg_c1 : tensor<64xf64>
      %mu = stablehlo.add %m1, %bb : tensor<64xf64>
      %r = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<1xf64>
      impulse.yield %s1#0, %ss_b : tensor<2xui64>, tensor<1xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<64xf64>, tensor<64xf64>, tensor<1xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

}
