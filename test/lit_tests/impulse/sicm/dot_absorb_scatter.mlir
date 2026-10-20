// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_dot_absorb_scatter" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func private @normal_sampler(%rng : tensor<2xui64>, %std : tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>) {
    %sample = stablehlo.abs %std : tensor<3xf64>
    return %rng, %sample : tensor<2xui64>, tensor<3xf64>
  }
  func.func private @normal_logpdf(%x : tensor<3xf64>, %std : tensor<3xf64>) -> tensor<3xf64> {
    %neg = stablehlo.negate %x : tensor<3xf64>
    return %neg : tensor<3xf64>
  }

  func.func private @normal_sampler3x2(%rng : tensor<2xui64>, %std : tensor<3x2xf64>) -> (tensor<2xui64>, tensor<3x2xf64>) {
    %sample = stablehlo.abs %std : tensor<3x2xf64>
    return %rng, %sample : tensor<2xui64>, tensor<3x2xf64>
  }
  func.func private @normal_logpdf3x2(%x : tensor<3x2xf64>, %std : tensor<3x2xf64>) -> tensor<3x2xf64> {
    %neg = stablehlo.negate %x : tensor<3x2xf64>
    return %neg : tensor<3x2xf64>
  }

  // CHECK-LABEL: func.func @absorb_scatter
  // CHECK-DAG: %[[NONE:.+]] = stablehlo.constant dense<-1> : tensor<8xi64>
  // CHECK-DAG: %[[ONE:.+]] = stablehlo.constant dense<1> : tensor<3x1xi64>
  // CHECK-DAG: %[[ZERO:.+]] = stablehlo.constant dense<0> : tensor<3x1xi64>
  // CHECK-DAG: %[[IDX:.+]] = stablehlo.constant dense<{{\[\[}}1], [4], [6]]> : tensor<3x1xi64>
  func.func @absorb_scatter(%rng : tensor<2xui64>, %A : tensor<5x8xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<3xf64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[A:.+]]: tensor<5x8xf64>):
    %res:8 = impulse.infer_region @model(%rng, %A) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<5x8xf64>):
      // CHECK: %[[X:.+]]:2 = impulse.sample_region
      %x:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<3xf64>):
        %v = stablehlo.abs %s : tensor<3xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<3xf64>
      } logpdf {
      ^bb0(%px: tensor<3xf64>, %s: tensor<3xf64>):
        %lp = stablehlo.negate %px : tensor<3xf64>
        impulse.yield %lp : tensor<3xf64>
      } <{address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
      %zeros = stablehlo.constant dense<0.0> : tensor<8xf64>
      %idx = stablehlo.constant dense<[[1], [4], [6]]> : tensor<3x1xi64>
      %scattered = "stablehlo.scatter"(%zeros, %idx, %x#1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
      ^bb0(%old: tensor<f64>, %new: tensor<f64>):
        stablehlo.return %new : tensor<f64>
      }) : (tensor<8xf64>, tensor<3x1xi64>, tensor<3xf64>) -> tensor<8xf64>
      // CHECK: %[[SLICE:.+]] = stablehlo.slice %[[IDX]] [0:3, 0:1] : (tensor<3x1xi64>) -> tensor<3x1xi64>
      // CHECK-NEXT: %[[SCALED:.+]] = stablehlo.multiply %[[SLICE]], %[[ONE]] : tensor<3x1xi64>
      // CHECK-NEXT: %[[LIN:.+]] = stablehlo.add %[[ZERO]], %[[SCALED]] : tensor<3x1xi64>
      // CHECK-NEXT: %[[GATHER:.+]] = "stablehlo.gather"(%[[A]], %[[LIN]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 5, 1>}> : (tensor<5x8xf64>, tensor<3x1xi64>) -> tensor<5x3xf64>
      // Without unique_indices only the last write to each position is kept.
      // CHECK-NEXT: %[[POS:.+]] = stablehlo.iota dim = 0 : tensor<3xi64>
      // CHECK-NEXT: %[[WIN:.+]] = "stablehlo.scatter"(%[[NONE]], %[[LIN]], %[[POS]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
      // CHECK-NEXT: ^bb0(%[[OLD:.+]]: tensor<i64>, %[[NEW:.+]]: tensor<i64>):
      // CHECK-NEXT: %[[MAX:.+]] = stablehlo.maximum %[[OLD]], %[[NEW]] : tensor<i64>
      // CHECK-NEXT: stablehlo.return %[[MAX]] : tensor<i64>
      // CHECK-NEXT: }) : (tensor<8xi64>, tensor<3x1xi64>, tensor<3xi64>) -> tensor<8xi64>
      // CHECK-NEXT: %[[WINAT:.+]] = "stablehlo.gather"(%[[WIN]], %[[LIN]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<8xi64>, tensor<3x1xi64>) -> tensor<3xi64>
      // CHECK-NEXT: %[[LAST:.+]] = stablehlo.compare EQ, %[[WINAT]], %[[POS]] : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
      // CHECK-NEXT: %[[KEEP:.+]] = stablehlo.convert %[[LAST]] : (tensor<3xi1>) -> tensor<3xf64>
      // CHECK-NEXT: %[[KEEPB:.+]] = stablehlo.broadcast_in_dim %[[KEEP]], dims = [1] : (tensor<3xf64>) -> tensor<5x3xf64>
      // CHECK-NEXT: %[[MASKED:.+]] = stablehlo.multiply %[[GATHER]], %[[KEEPB]] : tensor<5x3xf64>
      // CHECK-NEXT: %[[DOT:.+]] = stablehlo.dot_general %[[MASKED]], %[[X]]#1, contracting_dims = [1] x [0] : (tensor<5x3xf64>, tensor<3xf64>) -> tensor<5xf64>
      %y = stablehlo.dot_general %arg_A, %scattered, contracting_dims = [1] x [0] : (tensor<5x8xf64>, tensor<8xf64>) -> tensor<5xf64>
      // CHECK: impulse.yield %[[X]]#0, %[[DOT]] : tensor<2xui64>, tensor<5xf64>
      impulse.yield %x#0, %y : tensor<2xui64>, tensor<5xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<5x8xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @absorb_scatter_duplicates
  // CHECK-DAG: %[[IDX:.+]] = stablehlo.constant dense<{{\[\[}}1], [4], [1]]> : tensor<3x1xi64>
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[A:.+]]: tensor<5x8xf64>):
  // CHECK: %[[X:.+]]:2 = impulse.sample_region
  // CHECK: %[[GATHER:.+]] = "stablehlo.gather"(%[[A]], %[[LIN:[^)]+]]) <{{.+}}> : (tensor<5x8xf64>, tensor<3x1xi64>) -> tensor<5x3xf64>
  // CHECK-NEXT: %[[POS:.+]] = stablehlo.iota dim = 0 : tensor<3xi64>
  // CHECK-NEXT: %[[WIN:.+]] = "stablehlo.scatter"(%{{.+}}, %[[LIN]], %[[POS]])
  // CHECK: stablehlo.maximum
  // CHECK: %[[WINAT:.+]] = "stablehlo.gather"(%[[WIN]], %[[LIN]])
  // CHECK-NEXT: %[[LAST:.+]] = stablehlo.compare EQ, %[[WINAT]], %[[POS]] : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
  // CHECK-NEXT: %[[KEEP:.+]] = stablehlo.convert %[[LAST]] : (tensor<3xi1>) -> tensor<3xf64>
  // CHECK-NEXT: %[[KEEPB:.+]] = stablehlo.broadcast_in_dim %[[KEEP]], dims = [1] : (tensor<3xf64>) -> tensor<5x3xf64>
  // CHECK-NEXT: %[[MASKED:.+]] = stablehlo.multiply %[[GATHER]], %[[KEEPB]] : tensor<5x3xf64>
  // CHECK-NEXT: %[[DOT:.+]] = stablehlo.dot_general %[[MASKED]], %[[X]]#1, contracting_dims = [1] x [0] : (tensor<5x3xf64>, tensor<3xf64>) -> tensor<5xf64>
  // CHECK-NEXT: impulse.yield %[[X]]#0, %[[DOT]] : tensor<2xui64>, tensor<5xf64>
  func.func @absorb_scatter_duplicates(%rng : tensor<2xui64>, %A : tensor<5x8xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<3xf64>
    %res:8 = impulse.infer_region @model(%rng, %A) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<5x8xf64>):
      %x:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<3xf64>):
        %v = stablehlo.abs %s : tensor<3xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<3xf64>
      } logpdf {
      ^bb0(%px: tensor<3xf64>, %s: tensor<3xf64>):
        %lp = stablehlo.negate %px : tensor<3xf64>
        impulse.yield %lp : tensor<3xf64>
      } <{address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
      %zeros = stablehlo.constant dense<0.0> : tensor<8xf64>
      %idx = stablehlo.constant dense<[[1], [4], [1]]> : tensor<3x1xi64>
      %scattered = "stablehlo.scatter"(%zeros, %idx, %x#1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
      ^bb0(%old: tensor<f64>, %new: tensor<f64>):
        stablehlo.return %new : tensor<f64>
      }) : (tensor<8xf64>, tensor<3x1xi64>, tensor<3xf64>) -> tensor<8xf64>
      %y = stablehlo.dot_general %arg_A, %scattered, contracting_dims = [1] x [0] : (tensor<5x8xf64>, tensor<8xf64>) -> tensor<5xf64>
      impulse.yield %x#0, %y : tensor<2xui64>, tensor<5xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<5x8xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @absorb_scatter_unique
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[A:.+]]: tensor<5x8xf64>, %[[I:.+]]: tensor<3x1xi64>):
  // CHECK: %[[X:.+]]:2 = impulse.sample_region
  // CHECK: %[[SL:.+]] = stablehlo.slice %[[I]] [0:3, 0:1] : (tensor<3x1xi64>) -> tensor<3x1xi64>
  // CHECK-NEXT: %[[SC:.+]] = stablehlo.multiply %[[SL]], %{{.+}} : tensor<3x1xi64>
  // CHECK-NEXT: %[[LIN:.+]] = stablehlo.add %{{.+}}, %[[SC]] : tensor<3x1xi64>
  // CHECK-NEXT: %[[GATHER:.+]] = "stablehlo.gather"(%[[A]], %[[LIN]]) <{{.+}}> : (tensor<5x8xf64>, tensor<3x1xi64>) -> tensor<5x3xf64>
  // CHECK-NEXT: %[[DOT:.+]] = stablehlo.dot_general %[[GATHER]], %[[X]]#1, contracting_dims = [1] x [0] : (tensor<5x3xf64>, tensor<3xf64>) -> tensor<5xf64>
  // CHECK-NEXT: impulse.yield %[[X]]#0, %[[DOT]] : tensor<2xui64>, tensor<5xf64>
  func.func @absorb_scatter_unique(%rng : tensor<2xui64>, %A : tensor<5x8xf64>, %idx : tensor<3x1xi64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<3xf64>
    %res:8 = impulse.infer_region @model(%rng, %A, %idx) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<5x8xf64>, %arg_idx: tensor<3x1xi64>):
      %x:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<3xf64>):
        %v = stablehlo.abs %s : tensor<3xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<3xf64>
      } logpdf {
      ^bb0(%px: tensor<3xf64>, %s: tensor<3xf64>):
        %lp = stablehlo.negate %px : tensor<3xf64>
        impulse.yield %lp : tensor<3xf64>
      } <{address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
      %zeros = stablehlo.constant dense<0.0> : tensor<8xf64>
      %scattered = "stablehlo.scatter"(%zeros, %arg_idx, %x#1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
      ^bb0(%old: tensor<f64>, %new: tensor<f64>):
        stablehlo.return %new : tensor<f64>
      }) : (tensor<8xf64>, tensor<3x1xi64>, tensor<3xf64>) -> tensor<8xf64>
      %y = stablehlo.dot_general %arg_A, %scattered, contracting_dims = [1] x [0] : (tensor<5x8xf64>, tensor<8xf64>) -> tensor<5xf64>
      impulse.yield %x#0, %y : tensor<2xui64>, tensor<5xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<5x8xf64>, tensor<3x1xi64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @absorb_complex_pair
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[A:.+]]: tensor<5x8xcomplex<f64>>, %{{.+}}: tensor<3x1xi64>):
  // CHECK: %[[X:.+]]:2 = impulse.sample_region
  // CHECK: %[[XC:.+]] = stablehlo.complex %[[X]]#1, %{{.+}} : tensor<3xcomplex<f64>>
  // CHECK: %[[GATHER:.+]] = "stablehlo.gather"(%[[A]], %[[LIN:[^)]+]]) <{{.+}}> : (tensor<5x8xcomplex<f64>>, tensor<3x1xi64>) -> tensor<5x3xcomplex<f64>>
  // CHECK-NEXT: %[[POS:.+]] = stablehlo.iota dim = 0 : tensor<3xi64>
  // CHECK-NEXT: %[[WIN:.+]] = "stablehlo.scatter"(%{{.+}}, %[[LIN]], %[[POS]])
  // CHECK: stablehlo.maximum
  // CHECK: %[[WINAT:.+]] = "stablehlo.gather"(%[[WIN]], %[[LIN]])
  // CHECK-NEXT: %[[LAST:.+]] = stablehlo.compare EQ, %[[WINAT]], %[[POS]] : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
  // CHECK-NEXT: %[[KEEP:.+]] = stablehlo.convert %[[LAST]] : (tensor<3xi1>) -> tensor<3xcomplex<f64>>
  // CHECK-NEXT: %[[KEEPB:.+]] = stablehlo.broadcast_in_dim %[[KEEP]], dims = [1] : (tensor<3xcomplex<f64>>) -> tensor<5x3xcomplex<f64>>
  // CHECK-NEXT: %[[MASKED:.+]] = stablehlo.multiply %[[GATHER]], %[[KEEPB]] : tensor<5x3xcomplex<f64>>
  // CHECK-NEXT: %{{.+}} = stablehlo.dot_general %[[MASKED]], %[[XC]], contracting_dims = [1] x [0] : (tensor<5x3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> tensor<5xcomplex<f64>>
  func.func @absorb_complex_pair(%rng : tensor<2xui64>, %A : tensor<5x8xcomplex<f64>>, %idx : tensor<3x1xi64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<3xf64>
    %res:8 = impulse.infer_region @model(%rng, %A, %idx) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<5x8xcomplex<f64>>, %arg_idx: tensor<3x1xi64>):
      %x:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<3xf64>):
        %v = stablehlo.abs %s : tensor<3xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<3xf64>
      } logpdf {
      ^bb0(%px: tensor<3xf64>, %s: tensor<3xf64>):
        %lp = stablehlo.negate %px : tensor<3xf64>
        impulse.yield %lp : tensor<3xf64>
      } <{address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
      %zeros = stablehlo.constant dense<0.0> : tensor<8xf64>
      %neg = stablehlo.negate %x#1 : tensor<3xf64>
      %sre = "stablehlo.scatter"(%zeros, %arg_idx, %x#1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
      ^bb0(%old: tensor<f64>, %new: tensor<f64>):
        stablehlo.return %new : tensor<f64>
      }) : (tensor<8xf64>, tensor<3x1xi64>, tensor<3xf64>) -> tensor<8xf64>
      %sim = "stablehlo.scatter"(%zeros, %arg_idx, %neg) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
      ^bb0(%old: tensor<f64>, %new: tensor<f64>):
        stablehlo.return %new : tensor<f64>
      }) : (tensor<8xf64>, tensor<3x1xi64>, tensor<3xf64>) -> tensor<8xf64>
      %c = stablehlo.complex %sre, %sim : tensor<8xcomplex<f64>>
      %y = stablehlo.dot_general %arg_A, %c, contracting_dims = [1] x [0] : (tensor<5x8xcomplex<f64>>, tensor<8xcomplex<f64>>) -> tensor<5xcomplex<f64>>
      %yr = stablehlo.real %y : (tensor<5xcomplex<f64>>) -> tensor<5xf64>
      impulse.yield %x#0, %yr : tensor<2xui64>, tensor<5xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<5x8xcomplex<f64>>, tensor<3x1xi64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @absorb_channel_complex
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[A:.+]]: tensor<5x8xcomplex<f64>>, %{{.+}}: tensor<3x1xi64>):
  // CHECK: %[[X:.+]]:2 = impulse.sample_region
  // CHECK: %[[XC:.+]] = stablehlo.complex %{{.+}}, %{{.+}} : tensor<3xcomplex<f64>>
  // CHECK: %[[GATHER:.+]] = "stablehlo.gather"(%[[A]], %[[LIN:[^)]+]]) <{{.+}}> : (tensor<5x8xcomplex<f64>>, tensor<3x1xi64>) -> tensor<5x3xcomplex<f64>>
  // CHECK-NEXT: %[[POS:.+]] = stablehlo.iota dim = 0 : tensor<3xi64>
  // CHECK-NEXT: %[[WIN:.+]] = "stablehlo.scatter"(%{{.+}}, %[[LIN]], %[[POS]])
  // CHECK: stablehlo.maximum
  // CHECK: %[[WINAT:.+]] = "stablehlo.gather"(%[[WIN]], %[[LIN]])
  // CHECK-NEXT: %[[LAST:.+]] = stablehlo.compare EQ, %[[WINAT]], %[[POS]] : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
  // CHECK-NEXT: %[[KEEP:.+]] = stablehlo.convert %[[LAST]] : (tensor<3xi1>) -> tensor<3xcomplex<f64>>
  // CHECK-NEXT: %[[KEEPB:.+]] = stablehlo.broadcast_in_dim %[[KEEP]], dims = [1] : (tensor<3xcomplex<f64>>) -> tensor<5x3xcomplex<f64>>
  // CHECK-NEXT: %[[MASKED:.+]] = stablehlo.multiply %[[GATHER]], %[[KEEPB]] : tensor<5x3xcomplex<f64>>
  // CHECK-NEXT: %{{.+}} = stablehlo.dot_general %[[MASKED]], %[[XC]], contracting_dims = [1] x [0] : (tensor<5x3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> tensor<5xcomplex<f64>>
  func.func @absorb_channel_complex(%rng : tensor<2xui64>, %A : tensor<5x8xcomplex<f64>>, %idx : tensor<3x1xi64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<3x2xf64>
    %res:8 = impulse.infer_region @model(%rng, %A, %idx) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<5x8xcomplex<f64>>, %arg_idx: tensor<3x1xi64>):
      %x:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<3x2xf64>):
        %v = stablehlo.abs %s : tensor<3x2xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<3x2xf64>
      } logpdf {
      ^bb0(%px: tensor<3x2xf64>, %s: tensor<3x2xf64>):
        %lp = stablehlo.negate %px : tensor<3x2xf64>
        impulse.yield %lp : tensor<3x2xf64>
      } <{address = [#impulse.symbol<1>], fn = @normal_sampler3x2, logpdf_fn = @normal_logpdf3x2, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<3x2xf64>) -> (tensor<2xui64>, tensor<3x2xf64>)
      %zeros = stablehlo.constant dense<0.0> : tensor<8x2xf64>
      %s = "stablehlo.scatter"(%zeros, %arg_idx, %x#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
      ^bb0(%old: tensor<f64>, %new: tensor<f64>):
        stablehlo.return %new : tensor<f64>
      }) : (tensor<8x2xf64>, tensor<3x1xi64>, tensor<3x2xf64>) -> tensor<8x2xf64>
      %re2 = stablehlo.slice %s [0:8, 0:1] : (tensor<8x2xf64>) -> tensor<8x1xf64>
      %im2 = stablehlo.slice %s [0:8, 1:2] : (tensor<8x2xf64>) -> tensor<8x1xf64>
      %re = stablehlo.reshape %re2 : (tensor<8x1xf64>) -> tensor<8xf64>
      %im = stablehlo.reshape %im2 : (tensor<8x1xf64>) -> tensor<8xf64>
      %c = stablehlo.complex %re, %im : tensor<8xcomplex<f64>>
      %y = stablehlo.dot_general %arg_A, %c, contracting_dims = [1] x [0] : (tensor<5x8xcomplex<f64>>, tensor<8xcomplex<f64>>) -> tensor<5xcomplex<f64>>
      %yr = stablehlo.real %y : (tensor<5xcomplex<f64>>) -> tensor<5xf64>
      impulse.yield %x#0, %yr : tensor<2xui64>, tensor<5xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<5x8xcomplex<f64>>, tensor<3x1xi64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
