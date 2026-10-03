// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_gaussian_sufficient_stats" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func private @normal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %std : tensor<2xui64>, tensor<f64>
  }
  func.func private @normal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  // CHECK-LABEL: func.func @affine_slope_intercept
  // CHECK-DAG: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<64x1xf64>
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[C1:.+]]: tensor<64xf64>, %[[Y:.+]]: tensor<64xf64>):
  // CHECK: %[[S1:.+]]:2 = impulse.sample_region
  // CHECK: %[[S2:.+]]:2 = impulse.sample_region
  // CHECK: %[[C1COL:.+]] = stablehlo.reshape %[[C1]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[X:.+]] = stablehlo.concatenate %[[C1COL]], %[[ONES]], dim = 1 : (tensor<64x1xf64>, tensor<64x1xf64>) -> tensor<64x2xf64>
  // CHECK-NEXT: %[[BETA:.+]] = stablehlo.concatenate %[[S1]]#1, %[[S2]]#1, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  // CHECK-NEXT: %[[YCOL:.+]] = stablehlo.reshape %[[Y]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[AUG:.+]] = stablehlo.concatenate %[[X]], %[[YCOL]], dim = 1 : (tensor<64x2xf64>, tensor<64x1xf64>) -> tensor<64x3xf64>
  // CHECK-NEXT: %{{.+}}, %[[P:.+]], %{{.+}} = enzymexla.linalg.qr %[[AUG]] : (tensor<64x3xf64>) -> (tensor<64x3xf64>, tensor<64x3xf64>, tensor<i32>)
  // CHECK-NEXT: %[[RPACK:.+]] = stablehlo.slice %[[P]] [0:2, 0:2] : (tensor<64x3xf64>) -> tensor<2x2xf64>
  // CHECK-NEXT: %[[ROWS:.+]] = stablehlo.iota dim = 0 : tensor<2x2xi64>
  // CHECK-NEXT: %[[COLS:.+]] = stablehlo.iota dim = 1 : tensor<2x2xi64>
  // CHECK-NEXT: %[[UPPER:.+]] = stablehlo.compare LE, %[[ROWS]], %[[COLS]] : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi1>
  // CHECK-NEXT: %[[MASK:.+]] = stablehlo.convert %[[UPPER]] : (tensor<2x2xi1>) -> tensor<2x2xf64>
  // CHECK-NEXT: %[[RMAT:.+]] = stablehlo.multiply %[[RPACK]], %[[MASK]] : tensor<2x2xf64>
  // CHECK-NEXT: %[[CCOL:.+]] = stablehlo.slice %[[P]] [0:2, 2:3] : (tensor<64x3xf64>) -> tensor<2x1xf64>
  // CHECK-NEXT: %[[C:.+]] = stablehlo.reshape %[[CCOL]] : (tensor<2x1xf64>) -> tensor<2xf64>
  // CHECK-NEXT: %[[PDD:.+]] = stablehlo.slice %[[P]] [2:3, 2:3] : (tensor<64x3xf64>) -> tensor<1x1xf64>
  // CHECK-NEXT: %[[PD:.+]] = stablehlo.reshape %[[PDD]] : (tensor<1x1xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[REM:.+]] = stablehlo.multiply %[[PD]], %[[PD]] : tensor<f64>
  // CHECK-NEXT: %[[RB:.+]] = stablehlo.dot_general %[[RMAT]], %[[BETA]], contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
  // CHECK-NEXT: %[[D:.+]] = stablehlo.subtract %[[C]], %[[RB]] : tensor<2xf64>
  // CHECK-NEXT: %[[DD:.+]] = stablehlo.dot_general %[[D]], %[[D]], contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[RSS:.+]] = stablehlo.add %[[DD]], %[[REM]] : tensor<f64>
  // CHECK-NEXT: %{{.+}} = stablehlo.broadcast_in_dim %[[RSS]], dims = [] : (tensor<f64>) -> tensor<2xf64>
  // CHECK-NOT: stablehlo.multiply %{{.+}}, %[[C1]]
  func.func @affine_slope_intercept(%rng : tensor<2xui64>, %c1 : tensor<64xf64>, %y : tensor<64xf64>, %trace : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %c1, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_c1: tensor<64xf64>, %arg_y: tensor<64xf64>):
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
      %mu = stablehlo.add %m1, %b2 : tensor<64xf64>
      %r = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>

      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<2xf64>
      impulse.yield %s2#0, %ss_b : tensor<2xui64>, tensor<2xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]} : (tensor<2xui64>, tensor<64xf64>, tensor<64xf64>, tensor<1x2xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @affine_intercept_first
  // CHECK-DAG: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<64x1xf64>
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[C1:.+]]: tensor<64xf64>, %[[Y:.+]]: tensor<64xf64>):
  // CHECK: %[[S1:.+]]:2 = impulse.sample_region
  // CHECK: %[[S2:.+]]:2 = impulse.sample_region
  // CHECK: %[[C1COL:.+]] = stablehlo.reshape %[[C1]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[X:.+]] = stablehlo.concatenate %[[ONES]], %[[C1COL]], dim = 1 : (tensor<64x1xf64>, tensor<64x1xf64>) -> tensor<64x2xf64>
  // CHECK-NEXT: %[[BETA:.+]] = stablehlo.concatenate %[[S2]]#1, %[[S1]]#1, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  // CHECK-NEXT: %[[YCOL:.+]] = stablehlo.reshape %[[Y]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[AUG:.+]] = stablehlo.concatenate %[[X]], %[[YCOL]], dim = 1 : (tensor<64x2xf64>, tensor<64x1xf64>) -> tensor<64x3xf64>
  // CHECK-NEXT: %{{.+}}, %{{.+}}, %{{.+}} = enzymexla.linalg.qr %[[AUG]] : (tensor<64x3xf64>) -> (tensor<64x3xf64>, tensor<64x3xf64>, tensor<i32>)
  // CHECK: %[[RB:.+]] = stablehlo.dot_general %{{.+}}, %[[BETA]], contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
  // CHECK: %[[RSS:.+]] = stablehlo.add %{{.+}}, %{{.+}} : tensor<f64>
  // CHECK-NEXT: %[[SIG2:.+]] = stablehlo.multiply %[[SIG:.+]], %[[SIG]] : tensor<f64>
  // CHECK-NEXT: %{{.+}} = stablehlo.divide %[[RSS]], %[[SIG2]] : tensor<f64>
  // CHECK-NOT: stablehlo.reduce
  func.func @affine_intercept_first(%rng : tensor<2xui64>, %c1 : tensor<64xf64>, %y : tensor<64xf64>, %trace : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %c1, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_c1: tensor<64xf64>, %arg_y: tensor<64xf64>):
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
      %mu = stablehlo.add %b2, %m1 : tensor<64xf64>
      %cen = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %sig = stablehlo.constant dense<0.1> : tensor<f64>
      %sig_b = stablehlo.broadcast_in_dim %sig, dims = [] : (tensor<f64>) -> tensor<64xf64>
      %z = stablehlo.divide %cen, %sig_b : tensor<64xf64>
      %z2 = stablehlo.multiply %z, %z : tensor<64xf64>
      %log2pi = stablehlo.constant dense<1.8378770664093453> : tensor<64xf64>
      %sm = stablehlo.add %z2, %log2pi : tensor<64xf64>
      %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %ss = stablehlo.reduce(%sm init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<64xf64>, tensor<f64>) -> tensor<f64>

      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<2xf64>
      impulse.yield %s2#0, %ss_b : tensor<2xui64>, tensor<2xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]} : (tensor<2xui64>, tensor<64xf64>, tensor<64xf64>, tensor<1x2xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @affine_column_first
  // CHECK-DAG: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<64x1xf64>
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[C1:.+]]: tensor<64xf64>, %[[Y:.+]]: tensor<64xf64>):
  // CHECK: %[[S1:.+]]:2 = impulse.sample_region
  // CHECK: %[[S2:.+]]:2 = impulse.sample_region
  // CHECK: %[[C1COL:.+]] = stablehlo.reshape %[[C1]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[X:.+]] = stablehlo.concatenate %[[C1COL]], %[[ONES]], dim = 1 : (tensor<64x1xf64>, tensor<64x1xf64>) -> tensor<64x2xf64>
  // CHECK-NEXT: %[[BETA:.+]] = stablehlo.concatenate %[[S1]]#1, %[[S2]]#1, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  // CHECK-NEXT: %[[YCOL:.+]] = stablehlo.reshape %[[Y]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[AUG:.+]] = stablehlo.concatenate %[[X]], %[[YCOL]], dim = 1 : (tensor<64x2xf64>, tensor<64x1xf64>) -> tensor<64x3xf64>
  // CHECK-NEXT: %{{.+}}, %{{.+}}, %{{.+}} = enzymexla.linalg.qr %[[AUG]] : (tensor<64x3xf64>) -> (tensor<64x3xf64>, tensor<64x3xf64>, tensor<i32>)
  // CHECK: %{{.+}} = stablehlo.dot_general %{{.+}}, %[[BETA]], contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
  func.func @affine_column_first(%rng : tensor<2xui64>, %c1 : tensor<64xf64>, %y : tensor<64xf64>, %trace : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %c1, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_c1: tensor<64xf64>, %arg_y: tensor<64xf64>):
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
      %m1 = stablehlo.multiply %arg_c1, %b1 : tensor<64xf64>
      %b2 = stablehlo.broadcast_in_dim %s2#1, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %mu = stablehlo.add %m1, %b2 : tensor<64xf64>
      %r = stablehlo.subtract %mu, %arg_y : tensor<64xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>

      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<2xf64>
      impulse.yield %s2#0, %ss_b : tensor<2xui64>, tensor<2xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]} : (tensor<2xui64>, tensor<64xf64>, tensor<64xf64>, tensor<1x2xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @affine_subtract
  // CHECK-DAG: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<64x1xf64>
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[C1:.+]]: tensor<64xf64>, %[[Y:.+]]: tensor<64xf64>):
  // CHECK: %[[S1:.+]]:2 = impulse.sample_region
  // CHECK: %[[S2:.+]]:2 = impulse.sample_region
  // CHECK: %[[C1COL:.+]] = stablehlo.reshape %[[C1]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[NEGONES:.+]] = stablehlo.negate %[[ONES]] : tensor<64x1xf64>
  // CHECK-NEXT: %[[X:.+]] = stablehlo.concatenate %[[C1COL]], %[[NEGONES]], dim = 1 : (tensor<64x1xf64>, tensor<64x1xf64>) -> tensor<64x2xf64>
  // CHECK-NEXT: %[[BETA:.+]] = stablehlo.concatenate %[[S1]]#1, %[[S2]]#1, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  // CHECK-NEXT: %[[YCOL:.+]] = stablehlo.reshape %[[Y]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[AUG:.+]] = stablehlo.concatenate %[[X]], %[[YCOL]], dim = 1 : (tensor<64x2xf64>, tensor<64x1xf64>) -> tensor<64x3xf64>
  // CHECK-NEXT: %{{.+}}, %{{.+}}, %{{.+}} = enzymexla.linalg.qr %[[AUG]] : (tensor<64x3xf64>) -> (tensor<64x3xf64>, tensor<64x3xf64>, tensor<i32>)
  // CHECK: %{{.+}} = stablehlo.dot_general %{{.+}}, %[[BETA]], contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
  func.func @affine_subtract(%rng : tensor<2xui64>, %c1 : tensor<64xf64>, %y : tensor<64xf64>, %trace : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %c1, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_c1: tensor<64xf64>, %arg_y: tensor<64xf64>):
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
      %mu = stablehlo.subtract %m1, %b2 : tensor<64xf64>
      %r = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>

      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<2xf64>
      impulse.yield %s2#0, %ss_b : tensor<2xui64>, tensor<2xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]} : (tensor<2xui64>, tensor<64xf64>, tensor<64xf64>, tensor<1x2xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @affine_three_terms
  // CHECK-DAG: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<64x1xf64>
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[C1:.+]]: tensor<64xf64>, %[[Y:.+]]: tensor<64xf64>, %[[C2:.+]]: tensor<64xf64>):
  // CHECK: %[[S1:.+]]:2 = impulse.sample_region
  // CHECK: %[[S2:.+]]:2 = impulse.sample_region
  // CHECK: %[[S3:.+]]:2 = impulse.sample_region
  // CHECK: %[[C1COL:.+]] = stablehlo.reshape %[[C1]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[C2COL:.+]] = stablehlo.reshape %[[C2]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[X:.+]] = stablehlo.concatenate %[[C1COL]], %[[C2COL]], %[[ONES]], dim = 1 : (tensor<64x1xf64>, tensor<64x1xf64>, tensor<64x1xf64>) -> tensor<64x3xf64>
  // CHECK-NEXT: %[[BETA:.+]] = stablehlo.concatenate %[[S1]]#1, %[[S2]]#1, %[[S3]]#1, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[YCOL:.+]] = stablehlo.reshape %[[Y]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[AUG:.+]] = stablehlo.concatenate %[[X]], %[[YCOL]], dim = 1 : (tensor<64x3xf64>, tensor<64x1xf64>) -> tensor<64x4xf64>
  // CHECK-NEXT: %{{.+}}, %{{.+}}, %{{.+}} = enzymexla.linalg.qr %[[AUG]] : (tensor<64x4xf64>) -> (tensor<64x4xf64>, tensor<64x4xf64>, tensor<i32>)
  // CHECK: %{{.+}} = stablehlo.dot_general %{{.+}}, %[[BETA]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
  func.func @affine_three_terms(%rng : tensor<2xui64>, %c1 : tensor<64xf64>, %y : tensor<64xf64>, %c2 : tensor<64xf64>, %trace : tensor<1x3xf64>) -> (tensor<1x3xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %c1, %y, %c2) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_c1: tensor<64xf64>, %arg_y: tensor<64xf64>, %arg_c2 : tensor<64xf64>):
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
      %s3:2 = impulse.sample_region(%s2#0, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lx: tensor<1xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<3>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
      %b1 = stablehlo.broadcast_in_dim %s1#1, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %m1 = stablehlo.multiply %b1, %arg_c1 : tensor<64xf64>
      %b2 = stablehlo.broadcast_in_dim %s2#1, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %m2 = stablehlo.multiply %b2, %arg_c2 : tensor<64xf64>
      %b3 = stablehlo.broadcast_in_dim %s3#1, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %a1 = stablehlo.add %m1, %m2 : tensor<64xf64>
      %mu = stablehlo.add %a1, %b3 : tensor<64xf64>
      %r = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>

      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<3xf64>
      impulse.yield %s3#0, %ss_b : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]]} : (tensor<2xui64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<1x3xf64>, tensor<f64>) -> (tensor<1x3xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<f64>, tensor<f64>, tensor<1x3xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x3xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @affine_offset
  // CHECK-DAG: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<64x1xf64>
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[C1:.+]]: tensor<64xf64>, %[[Y:.+]]: tensor<64xf64>, %[[O:.+]]: tensor<64xf64>):
  // CHECK: %[[S1:.+]]:2 = impulse.sample_region
  // CHECK: %[[S2:.+]]:2 = impulse.sample_region
  // CHECK: %[[C1COL:.+]] = stablehlo.reshape %[[C1]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[X:.+]] = stablehlo.concatenate %[[C1COL]], %[[ONES]], dim = 1 : (tensor<64x1xf64>, tensor<64x1xf64>) -> tensor<64x2xf64>
  // CHECK-NEXT: %[[BETA:.+]] = stablehlo.concatenate %[[S1]]#1, %[[S2]]#1, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  // CHECK-NEXT: %[[YP:.+]] = stablehlo.subtract %[[Y]], %[[O]] : tensor<64xf64>
  // CHECK-NEXT: %[[YCOL:.+]] = stablehlo.reshape %[[YP]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[AUG:.+]] = stablehlo.concatenate %[[X]], %[[YCOL]], dim = 1 : (tensor<64x2xf64>, tensor<64x1xf64>) -> tensor<64x3xf64>
  // CHECK-NEXT: %{{.+}}, %{{.+}}, %{{.+}} = enzymexla.linalg.qr %[[AUG]] : (tensor<64x3xf64>) -> (tensor<64x3xf64>, tensor<64x3xf64>, tensor<i32>)
  func.func @affine_offset(%rng : tensor<2xui64>, %c1 : tensor<64xf64>, %y : tensor<64xf64>, %o : tensor<64xf64>, %trace : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %c1, %y, %o) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_c1: tensor<64xf64>, %arg_y: tensor<64xf64>, %arg_o : tensor<64xf64>):
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
      %a1 = stablehlo.add %m1, %b2 : tensor<64xf64>
      %mu = stablehlo.add %a1, %arg_o : tensor<64xf64>
      %r = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>

      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<2xf64>
      impulse.yield %s2#0, %ss_b : tensor<2xui64>, tensor<2xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]} : (tensor<2xui64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<1x2xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @neg_dependent_vector
  // CHECK-NOT: enzymexla.linalg.qr
  // CHECK: stablehlo.dot_general %[[R:.+]], %[[R]], contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
  func.func @neg_dependent_vector(%rng : tensor<2xui64>, %c1 : tensor<64xf64>, %y : tensor<64xf64>, %trace : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %c1, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_c1: tensor<64xf64>, %arg_y: tensor<64xf64>):
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
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<64xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<64xf64>
      } logpdf {
      ^bb0(%lx: tensor<64xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<2>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<64xf64>)
      %b1 = stablehlo.broadcast_in_dim %s1#1, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %m1 = stablehlo.multiply %b1, %arg_c1 : tensor<64xf64>
      %mu = stablehlo.add %m1, %s2#1 : tensor<64xf64>
      %r = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>

      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<2xf64>
      impulse.yield %s2#0, %ss_b : tensor<2xui64>, tensor<2xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]} : (tensor<2xui64>, tensor<64xf64>, tensor<64xf64>, tensor<1x2xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

}
