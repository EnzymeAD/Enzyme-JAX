// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_log_multiply;impulse.sicm_dot_general_scale" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// log(f(s * L)) = log(s) + log(f(L)) for f homogeneous of degree one in L.

module {
  func.func private @halfnormal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %sample = stablehlo.abs %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }
  func.func private @halfnormal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  // CHECK-LABEL: func.func @strided_slice_diagonal
  func.func @strided_slice_diagonal(%rng : tensor<2xui64>, %L : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[L:.+]]: tensor<3x3xf64>):
    %res:8 = impulse.infer_region @model(%rng, %L) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_L: tensor<3x3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
      // CHECK: %[[TAUSQ:.+]] = stablehlo.multiply %[[TAU]]#1, %[[TAU]]#1 : tensor<f64>
      // CHECK-NEXT: %[[SIG:.+]] = stablehlo.sqrt %[[TAUSQ]]{{( \{.+\})?}} : tensor<f64>
      // CHECK-NEXT: %[[TR:.+]] = stablehlo.transpose %[[L]], dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
      // CHECK-NEXT: %[[FLAT:.+]] = stablehlo.reshape %[[TR]] : (tensor<3x3xf64>) -> tensor<9xf64>
      // CHECK-NEXT: %[[DIAG:.+]] = stablehlo.slice %[[FLAT]] [0:9:4] : (tensor<9xf64>) -> tensor<3xf64>
      // CHECK-NEXT: %[[LOGDIAG:.+]] = stablehlo.log %[[DIAG]] : tensor<3xf64>
      // CHECK-NEXT: %[[LOGS:.+]] = stablehlo.log %[[SIG]] : tensor<f64>
      // CHECK-NEXT: %[[LOGSB:.+]] = stablehlo.broadcast_in_dim %[[LOGS]], dims = [] : (tensor<f64>) -> tensor<3xf64>
      // CHECK-NEXT: %[[OUT:.+]] = stablehlo.add %[[LOGSB]], %[[LOGDIAG]] : tensor<3xf64>
      // CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[OUT]] : tensor<2xui64>, tensor<3xf64>
      %tausq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
      %sig = stablehlo.sqrt %tausq : tensor<f64>
      %bcast = stablehlo.broadcast_in_dim %sig, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %sL = stablehlo.multiply %bcast, %arg_L : tensor<3x3xf64>
      %tr = stablehlo.transpose %sL, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
      %flat = stablehlo.reshape %tr : (tensor<3x3xf64>) -> tensor<9xf64>
      %diag = stablehlo.slice %flat [0:9:4] : (tensor<9xf64>) -> tensor<3xf64>
      %logdiag = stablehlo.log %diag : tensor<3xf64>
      impulse.yield %tau#0, %logdiag : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @identity_contraction_diagonal
  func.func @identity_contraction_diagonal(%rng : tensor<2xui64>, %L : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %I = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[L:.+]]: tensor<3x3xf64>):
    %res:8 = impulse.infer_region @model(%rng, %L) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_L: tensor<3x3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
      // CHECK: %[[TAUSQ:.+]] = stablehlo.multiply %[[TAU]]#1, %[[TAU]]#1 : tensor<f64>
      // CHECK-NEXT: %[[SIG:.+]] = stablehlo.sqrt %[[TAUSQ]]{{( \{.+\})?}} : tensor<f64>
      // CHECK-NEXT: %[[DIAG:.+]] = stablehlo.dot_general %[[L]], %[[I:.+]], batching_dims = [1] x [1], contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3xf64>
      // CHECK-NEXT: %[[LOGDIAG:.+]] = stablehlo.log %[[DIAG]] : tensor<3xf64>
      // CHECK-NEXT: %[[LOGS:.+]] = stablehlo.log %[[SIG]] : tensor<f64>
      // CHECK-NEXT: %[[LOGSB:.+]] = stablehlo.broadcast_in_dim %[[LOGS]], dims = [] : (tensor<f64>) -> tensor<3xf64>
      // CHECK-NEXT: %[[OUT:.+]] = stablehlo.add %[[LOGSB]], %[[LOGDIAG]] : tensor<3xf64>
      // CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[OUT]] : tensor<2xui64>, tensor<3xf64>
      %tausq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
      %sig = stablehlo.sqrt %tausq : tensor<f64>
      %bcast = stablehlo.broadcast_in_dim %sig, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %sL = stablehlo.multiply %bcast, %arg_L : tensor<3x3xf64>
      %diag = stablehlo.dot_general %sL, %I, batching_dims = [1] x [1], contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3xf64>
      %logdiag = stablehlo.log %diag : tensor<3xf64>
      impulse.yield %tau#0, %logdiag : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @negative_non_selecting_op
  func.func @negative_non_selecting_op(%rng : tensor<2xui64>, %L : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %L) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_L: tensor<3x3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %tausq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
      %sig = stablehlo.sqrt %tausq : tensor<f64>
      %bcast = stablehlo.broadcast_in_dim %sig, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %sL = stablehlo.multiply %bcast, %arg_L : tensor<3x3xf64>
      // CHECK: %[[SL:.+]] = stablehlo.multiply %{{.+}}, %{{.+}} : tensor<3x3xf64>
      // CHECK-NEXT: %[[ABS:.+]] = stablehlo.abs %[[SL]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[FLAT:.+]] = stablehlo.reshape %[[ABS]] : (tensor<3x3xf64>) -> tensor<9xf64>
      // CHECK-NEXT: %[[DIAG:.+]] = stablehlo.slice %[[FLAT]] [0:9:4] : (tensor<9xf64>) -> tensor<3xf64>
      // CHECK-NEXT: %{{.+}} = stablehlo.log %[[DIAG]] : tensor<3xf64>
      %abs = stablehlo.abs %sL : tensor<3x3xf64>
      %flat = stablehlo.reshape %abs : (tensor<3x3xf64>) -> tensor<9xf64>
      %diag = stablehlo.slice %flat [0:9:4] : (tensor<9xf64>) -> tensor<3xf64>
      %logdiag = stablehlo.log %diag : tensor<3xf64>
      impulse.yield %tau#0, %logdiag : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @masked_diagonal
  func.func @masked_diagonal(%rng : tensor<2xui64>, %L : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %I = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    %ones = stablehlo.constant dense<1.0> : tensor<3xf64>
    // CHECK-DAG: %[[I:.+]] = stablehlo.constant dense<{{\[\[}}1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<3x3xf64>
    // CHECK-DAG: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[L:.+]]: tensor<3x3xf64>):
    %res:8 = impulse.infer_region @model(%rng, %L) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_L: tensor<3x3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
      // CHECK: %[[TAUSQ:.+]] = stablehlo.multiply %[[TAU]]#1, %[[TAU]]#1 : tensor<f64>
      // CHECK-NEXT: %[[SIG:.+]] = stablehlo.sqrt %[[TAUSQ]]{{( \{.+\})?}} : tensor<f64>
      // CHECK-NEXT: %[[MASKED:.+]] = stablehlo.multiply %[[L]], %[[I]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[DIAG:.+]] = stablehlo.dot_general %[[MASKED]], %[[ONES]], contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
      // CHECK-NEXT: %[[LOGDIAG:.+]] = stablehlo.log %[[DIAG]] : tensor<3xf64>
      // CHECK-NEXT: %[[LOGS:.+]] = stablehlo.log %[[SIG]] : tensor<f64>
      // CHECK-NEXT: %[[LOGSB:.+]] = stablehlo.broadcast_in_dim %[[LOGS]], dims = [] : (tensor<f64>) -> tensor<3xf64>
      // CHECK-NEXT: %[[OUT:.+]] = stablehlo.add %[[LOGSB]], %[[LOGDIAG]] : tensor<3xf64>
      // CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[OUT]] : tensor<2xui64>, tensor<3xf64>
      %tausq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
      %sig = stablehlo.sqrt %tausq : tensor<f64>
      %bcast = stablehlo.broadcast_in_dim %sig, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %sL = stablehlo.multiply %bcast, %arg_L : tensor<3x3xf64>
      %masked = stablehlo.multiply %sL, %I : tensor<3x3xf64>
      %diag = stablehlo.dot_general %masked, %ones, contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
      %logdiag = stablehlo.log %diag : tensor<3xf64>
      impulse.yield %tau#0, %logdiag : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @negative_dependent_mask
  func.func @negative_dependent_mask(%rng : tensor<2xui64>, %L : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %I = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    %ones = stablehlo.constant dense<1.0> : tensor<3xf64>
    // CHECK-DAG: %[[I:.+]] = stablehlo.constant dense<{{\[\[}}1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<3x3xf64>
    // CHECK-DAG: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[L:.+]]: tensor<3x3xf64>):
    %res:8 = impulse.infer_region @model(%rng, %L) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_L: tensor<3x3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
      // CHECK: %[[SL:.+]] = stablehlo.multiply %{{.+}}, %[[L]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[MB:.+]] = stablehlo.broadcast_in_dim %[[TAU]]#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      // CHECK-NEXT: %[[MASK:.+]] = stablehlo.multiply %[[MB]], %[[I]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[MASKED:.+]] = stablehlo.multiply %[[SL]], %[[MASK]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[DIAG:.+]] = stablehlo.dot_general %[[MASKED]], %[[ONES]], contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
      // CHECK-NEXT: %[[LOGDIAG:.+]] = stablehlo.log %[[DIAG]] : tensor<3xf64>
      // CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[LOGDIAG]] : tensor<2xui64>, tensor<3xf64>
      %tausq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
      %sig = stablehlo.sqrt %tausq : tensor<f64>
      %bcast = stablehlo.broadcast_in_dim %sig, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %sL = stablehlo.multiply %bcast, %arg_L : tensor<3x3xf64>
      %mb = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %mask = stablehlo.multiply %mb, %I : tensor<3x3xf64>
      %masked = stablehlo.multiply %sL, %mask : tensor<3x3xf64>
      %diag = stablehlo.dot_general %masked, %ones, contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
      %logdiag = stablehlo.log %diag : tensor<3xf64>
      impulse.yield %tau#0, %logdiag : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @negative_dependent_contraction
  func.func @negative_dependent_contraction(%rng : tensor<2xui64>, %L : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %I = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    // CHECK-DAG: %[[I:.+]] = stablehlo.constant dense<{{\[\[}}1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<3x3xf64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[L:.+]]: tensor<3x3xf64>):
    %res:8 = impulse.infer_region @model(%rng, %L) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_L: tensor<3x3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
      // CHECK: %[[SL:.+]] = stablehlo.multiply %{{.+}}, %[[L]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[MASKED:.+]] = stablehlo.multiply %[[SL]], %[[I]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[W:.+]] = stablehlo.broadcast_in_dim %[[TAU]]#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
      // CHECK-NEXT: %[[DIAG:.+]] = stablehlo.dot_general %[[MASKED]], %[[W]], contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
      // CHECK-NEXT: %[[LOGDIAG:.+]] = stablehlo.log %[[DIAG]] : tensor<3xf64>
      // CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[LOGDIAG]] : tensor<2xui64>, tensor<3xf64>
      %tausq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
      %sig = stablehlo.sqrt %tausq : tensor<f64>
      %bcast = stablehlo.broadcast_in_dim %sig, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %sL = stablehlo.multiply %bcast, %arg_L : tensor<3x3xf64>
      %masked = stablehlo.multiply %sL, %I : tensor<3x3xf64>
      %w = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %diag = stablehlo.dot_general %masked, %w, contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
      %logdiag = stablehlo.log %diag : tensor<3xf64>
      impulse.yield %tau#0, %logdiag : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @negative_unsigned_scale_masked
  func.func @negative_unsigned_scale_masked(%rng : tensor<2xui64>, %L : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %I = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    %ones = stablehlo.constant dense<1.0> : tensor<3xf64>
    // CHECK-DAG: %[[I:.+]] = stablehlo.constant dense<{{\[\[}}1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<3x3xf64>
    // CHECK-DAG: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[L:.+]]: tensor<3x3xf64>):
    %res:8 = impulse.infer_region @model(%rng, %L) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_L: tensor<3x3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
      // CHECK: %[[BC:.+]] = stablehlo.broadcast_in_dim %[[TAU]]#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      // CHECK-NEXT: %[[SL:.+]] = stablehlo.multiply %[[BC]], %[[L]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[MASKED:.+]] = stablehlo.multiply %[[SL]], %[[I]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[DIAG:.+]] = stablehlo.dot_general %[[MASKED]], %[[ONES]], contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
      // CHECK-NEXT: %[[LOGDIAG:.+]] = stablehlo.log %[[DIAG]] : tensor<3xf64>
      // CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[LOGDIAG]] : tensor<2xui64>, tensor<3xf64>
      %bcast = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %sL = stablehlo.multiply %bcast, %arg_L : tensor<3x3xf64>
      %masked = stablehlo.multiply %sL, %I : tensor<3x3xf64>
      %diag = stablehlo.dot_general %masked, %ones, contracting_dims = [0] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
      %logdiag = stablehlo.log %diag : tensor<3xf64>
      impulse.yield %tau#0, %logdiag : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
