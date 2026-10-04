// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_dot_general_shift" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func private @halfnormal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %sample = stablehlo.abs %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }
  func.func private @halfnormal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }
  func.func private @normal_sampler(%rng : tensor<2xui64>, %std : tensor<3x1xf64>) -> (tensor<2xui64>, tensor<3x1xf64>) {
    %sample = stablehlo.abs %std : tensor<3x1xf64>
    return %rng, %sample : tensor<2xui64>, tensor<3x1xf64>
  }
  func.func private @normal_logpdf(%x : tensor<3x1xf64>, %std : tensor<3x1xf64>) -> tensor<3x1xf64> {
    %neg = stablehlo.negate %x : tensor<3x1xf64>
    return %neg : tensor<3x1xf64>
  }

  // CHECK-LABEL: func.func @shift_lhs
  func.func @shift_lhs(%rng : tensor<2xui64>, %R : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %stdv = stablehlo.constant dense<1.0> : tensor<3x1xf64>
    %I = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[R:.+]]: tensor<3x3xf64>):
    %res:8 = impulse.infer_region @model(%rng, %R) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_R: tensor<3x3xf64>):
      %s:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %sd: tensor<f64>):
        %v = stablehlo.abs %sd : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%lx: tensor<f64>, %sd: tensor<f64>):
        %lp = stablehlo.negate %lx : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %t:2 = impulse.sample_region(%s#0, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %sd: tensor<f64>):
        %v = stablehlo.abs %sd : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%lx: tensor<f64>, %sd: tensor<f64>):
        %lp = stablehlo.negate %lx : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<2>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %phi:2 = impulse.sample_region(%t#0, %stdv) sampler {
      ^bb0(%r: tensor<2xui64>, %sd: tensor<3x1xf64>):
        %v = stablehlo.abs %sd : tensor<3x1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<3x1xf64>
      } logpdf {
      ^bb0(%lx: tensor<3x1xf64>, %sd: tensor<3x1xf64>):
        %lp = stablehlo.negate %lx : tensor<3x1xf64>
        impulse.yield %lp : tensor<3x1xf64>
      } {address = [#impulse.symbol<3>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<3x1xf64>) -> (tensor<2xui64>, tensor<3x1xf64>)
      // CHECK: %[[S:.+]]:2 = impulse.sample_region(%{{.+}}, %{{.+}}) sampler
      // CHECK: %[[T:.+]]:2 = impulse.sample_region(%[[S]]#0, %{{.+}}) sampler
      // CHECK: %[[PHI:.+]]:2 = impulse.sample_region(%[[T]]#0, %{{.+}}) sampler
      // CHECK: %[[S2:.+]] = stablehlo.reshape %[[S]]#1 : (tensor<f64>) -> tensor<1x1xf64>
      // CHECK-NEXT: %[[T2:.+]] = stablehlo.reshape %[[T]]#1 : (tensor<f64>) -> tensor<1x1xf64>
      // CHECK-NEXT: %[[RPHI:.+]] = stablehlo.dot_general %[[R]], %[[PHI]]#1, contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
      // CHECK-NEXT: %[[SS:.+]] = stablehlo.reshape %[[S2]] : (tensor<1x1xf64>) -> tensor<f64>
      // CHECK-NEXT: %[[SB:.+]] = stablehlo.broadcast_in_dim %[[SS]], dims = [] : (tensor<f64>) -> tensor<3x1xf64>
      // CHECK-NEXT: %[[SRPHI:.+]] = stablehlo.multiply %[[SB]], %[[RPHI]] : tensor<3x1xf64>
      // CHECK-NEXT: %[[TS:.+]] = stablehlo.reshape %[[T2]] : (tensor<1x1xf64>) -> tensor<f64>
      // CHECK-NEXT: %[[TB:.+]] = stablehlo.broadcast_in_dim %[[TS]], dims = [] : (tensor<f64>) -> tensor<3x1xf64>
      // CHECK-NEXT: %[[TPHI:.+]] = stablehlo.multiply %[[TB]], %[[PHI]]#1 : tensor<3x1xf64>
      // CHECK-NEXT: %[[Y:.+]] = stablehlo.add %[[SRPHI]], %[[TPHI]] : tensor<3x1xf64>
      // CHECK-NEXT: impulse.yield %[[PHI]]#0, %[[Y]] : tensor<2xui64>, tensor<3x1xf64>
      %s2 = stablehlo.reshape %s#1 : (tensor<f64>) -> tensor<1x1xf64>
      %t2 = stablehlo.reshape %t#1 : (tensor<f64>) -> tensor<1x1xf64>
      %sb = stablehlo.broadcast_in_dim %s2, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<3x3xf64>
      %sR = stablehlo.multiply %sb, %arg_R : tensor<3x3xf64>
      %tb = stablehlo.broadcast_in_dim %t2, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<3x3xf64>
      %tI = stablehlo.multiply %tb, %I : tensor<3x3xf64>
      %P = stablehlo.add %sR, %tI : tensor<3x3xf64>
      %y = stablehlo.dot_general %P, %phi#1, contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
      impulse.yield %phi#0, %y : tensor<2xui64>, tensor<3x1xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]]} : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // The identity term of x^T (A + t*I) is x transposed.
  // CHECK-LABEL: func.func @shift_rhs_factored
  func.func @shift_rhs_factored(%rng : tensor<2xui64>, %A : tensor<3x3xf64>, %x : tensor<3x2xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %J = stablehlo.constant dense<[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]> : tensor<3x3xf64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[A:.+]]: tensor<3x3xf64>, %[[X:.+]]: tensor<3x2xf64>):
    %res:8 = impulse.infer_region @model(%rng, %A, %x) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<3x3xf64>, %arg_x: tensor<3x2xf64>):
      %t:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %sd: tensor<f64>):
        %v = stablehlo.abs %sd : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%lx: tensor<f64>, %sd: tensor<f64>):
        %lp = stablehlo.negate %lx : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[T:.+]]:2 = impulse.sample_region
      // CHECK: %[[CT:.+]] = stablehlo.multiply %[[T]]#1, %[[HALF:.+]] : tensor<f64>
      // CHECK-NEXT: %[[XA:.+]] = stablehlo.dot_general %[[X]], %[[A]], contracting_dims = [0] x [0] : (tensor<3x2xf64>, tensor<3x3xf64>) -> tensor<2x3xf64>
      // CHECK-NEXT: %[[TB:.+]] = stablehlo.broadcast_in_dim %[[T]]#1, dims = [] : (tensor<f64>) -> tensor<2x3xf64>
      // CHECK-NEXT: %[[TXA:.+]] = stablehlo.multiply %[[TB]], %[[XA]] : tensor<2x3xf64>
      // CHECK-NEXT: %[[XT:.+]] = stablehlo.transpose %[[X]], dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
      // CHECK-NEXT: %[[CTB:.+]] = stablehlo.broadcast_in_dim %[[CT]], dims = [] : (tensor<f64>) -> tensor<2x3xf64>
      // CHECK-NEXT: %[[CTX:.+]] = stablehlo.multiply %[[CTB]], %[[XT]] : tensor<2x3xf64>
      // CHECK-NEXT: %[[Y:.+]] = stablehlo.add %[[TXA]], %[[CTX]] : tensor<2x3xf64>
      // CHECK-NEXT: impulse.yield %[[T]]#0, %[[Y]] : tensor<2xui64>, tensor<2x3xf64>
      %AJ = stablehlo.add %arg_A, %J : tensor<3x3xf64>
      %tb = stablehlo.broadcast_in_dim %t#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %P = stablehlo.multiply %tb, %AJ : tensor<3x3xf64>
      %y = stablehlo.dot_general %arg_x, %P, contracting_dims = [0] x [0] : (tensor<3x2xf64>, tensor<3x3xf64>) -> tensor<2x3xf64>
      impulse.yield %t#0, %y : tensor<2xui64>, tensor<2x3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<3x3xf64>, tensor<3x2xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @negative_non_identity_shift
  func.func @negative_non_identity_shift(%rng : tensor<2xui64>, %A : tensor<3x3xf64>, %B : tensor<3x3xf64>, %x : tensor<3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %A, %B, %x) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<3x3xf64>, %arg_B: tensor<3x3xf64>, %arg_x: tensor<3xf64>):
      %t:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %sd: tensor<f64>):
        %v = stablehlo.abs %sd : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%lx: tensor<f64>, %sd: tensor<f64>):
        %lp = stablehlo.negate %lx : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[TB:.+]] = stablehlo.broadcast_in_dim %{{.+}}#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      // CHECK-NEXT: %[[TBM:.+]] = stablehlo.multiply %[[TB]], %[[B:.+]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[P:.+]] = stablehlo.add %{{.+}}, %[[TBM]] : tensor<3x3xf64>
      // CHECK-NEXT: %{{.+}} = stablehlo.dot_general %[[P]], %{{.+}}, contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
      %tb = stablehlo.broadcast_in_dim %t#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %tB = stablehlo.multiply %tb, %arg_B : tensor<3x3xf64>
      %P = stablehlo.add %arg_A, %tB : tensor<3x3xf64>
      %y = stablehlo.dot_general %P, %arg_x, contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
      impulse.yield %t#0, %y : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<3x3xf64>, tensor<3x3xf64>, tensor<3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @negative_invariant_or_batched
  func.func @negative_invariant_or_batched(%rng : tensor<2xui64>, %A : tensor<3x3xf64>, %x : tensor<3xf64>, %z : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %I = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    %two = stablehlo.constant dense<2.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %A, %x, %z) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<3x3xf64>, %arg_x: tensor<3xf64>, %arg_z: tensor<3x3xf64>):
      %t:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %sd: tensor<f64>):
        %v = stablehlo.abs %sd : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%lx: tensor<f64>, %sd: tensor<f64>):
        %lp = stablehlo.negate %lx : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[PC:.+]] = stablehlo.add %{{.+}}, %{{.+}} : tensor<3x3xf64>
      // CHECK-NEXT: %{{.+}} = stablehlo.dot_general %[[PC]], %{{.+}}, contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
      // CHECK: %[[P:.+]] = stablehlo.add %{{.+}}, %{{.+}} : tensor<3x3xf64>
      // CHECK-NEXT: %{{.+}} = stablehlo.dot_general %[[P]], %{{.+}}, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3xf64>
      %twob = stablehlo.broadcast_in_dim %two, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %twoI = stablehlo.multiply %twob, %I : tensor<3x3xf64>
      %Pc = stablehlo.add %arg_A, %twoI : tensor<3x3xf64>
      %y0 = stablehlo.dot_general %Pc, %arg_x, contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
      %tb = stablehlo.broadcast_in_dim %t#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %tI = stablehlo.multiply %tb, %I : tensor<3x3xf64>
      %P = stablehlo.add %arg_A, %tI : tensor<3x3xf64>
      %y1 = stablehlo.dot_general %P, %arg_z, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3xf64>
      %y = stablehlo.add %y0, %y1 : tensor<3xf64>
      impulse.yield %t#0, %y : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<3x3xf64>, tensor<3xf64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
