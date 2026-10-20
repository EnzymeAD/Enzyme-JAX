// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_gaussian_sufficient_stats" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func private @normal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %std : tensor<2xui64>, tensor<f64>
  }
  func.func private @normal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  // CHECK-LABEL: func.func @intercept_plus_dot
  // CHECK-DAG: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<64x1xf64>
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[XC:.+]]: tensor<64x3xf64>, %[[Y:.+]]: tensor<64xf64>):
  // CHECK: %[[INT:.+]]:2 = impulse.sample_region
  // CHECK: %[[B:.+]]:2 = impulse.sample_region
  // CHECK: %[[SIG:.+]]:2 = impulse.sample_region
  // CHECK: %[[SIGS:.+]] = stablehlo.reshape %[[SIG]]#1 : (tensor<1xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[X:.+]] = stablehlo.concatenate %[[ONES]], %[[XC]], dim = 1 : (tensor<64x1xf64>, tensor<64x3xf64>) -> tensor<64x4xf64>
  // CHECK-NEXT: %[[BETA:.+]] = stablehlo.concatenate %[[INT]]#1, %[[B]]#1, dim = 0 : (tensor<1xf64>, tensor<3xf64>) -> tensor<4xf64>
  // CHECK-NEXT: %[[YCOL:.+]] = stablehlo.reshape %[[Y]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[AUG:.+]] = stablehlo.concatenate %[[X]], %[[YCOL]], dim = 1 : (tensor<64x4xf64>, tensor<64x1xf64>) -> tensor<64x5xf64>
  // CHECK-NEXT: %{{.+}}, %[[P:.+]], %{{.+}} = enzymexla.linalg.qr %[[AUG]] : (tensor<64x5xf64>) -> (tensor<64x5xf64>, tensor<64x5xf64>, tensor<i32>)
  // CHECK-NEXT: %[[RPACK:.+]] = stablehlo.slice %[[P]] [0:4, 0:4] : (tensor<64x5xf64>) -> tensor<4x4xf64>
  // CHECK-NEXT: %[[ROWS:.+]] = stablehlo.iota dim = 0 : tensor<4x4xi64>
  // CHECK-NEXT: %[[COLS:.+]] = stablehlo.iota dim = 1 : tensor<4x4xi64>
  // CHECK-NEXT: %[[UPPER:.+]] = stablehlo.compare LE, %[[ROWS]], %[[COLS]] : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1>
  // CHECK-NEXT: %[[MASK:.+]] = stablehlo.convert %[[UPPER]] : (tensor<4x4xi1>) -> tensor<4x4xf64>
  // CHECK-NEXT: %[[RMAT:.+]] = stablehlo.multiply %[[RPACK]], %[[MASK]] : tensor<4x4xf64>
  // CHECK-NEXT: %[[CCOL:.+]] = stablehlo.slice %[[P]] [0:4, 4:5] : (tensor<64x5xf64>) -> tensor<4x1xf64>
  // CHECK-NEXT: %[[C:.+]] = stablehlo.reshape %[[CCOL]] : (tensor<4x1xf64>) -> tensor<4xf64>
  // CHECK-NEXT: %[[PDD:.+]] = stablehlo.slice %[[P]] [4:5, 4:5] : (tensor<64x5xf64>) -> tensor<1x1xf64>
  // CHECK-NEXT: %[[PD:.+]] = stablehlo.reshape %[[PDD]] : (tensor<1x1xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[REM:.+]] = stablehlo.multiply %[[PD]], %[[PD]] : tensor<f64>
  // CHECK-NEXT: %[[RB:.+]] = stablehlo.dot_general %[[RMAT]], %[[BETA]], contracting_dims = [1] x [0] : (tensor<4x4xf64>, tensor<4xf64>) -> tensor<4xf64>
  // CHECK-NEXT: %[[D:.+]] = stablehlo.subtract %[[C]], %[[RB]] : tensor<4xf64>
  // CHECK-NEXT: %[[DD:.+]] = stablehlo.dot_general %[[D]], %[[D]], contracting_dims = [0] x [0] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[RSS:.+]] = stablehlo.add %[[DD]], %[[REM]] : tensor<f64>
  // CHECK-NEXT: %[[SIG2:.+]] = stablehlo.multiply %[[SIGS]], %[[SIGS]] : tensor<f64>
  // CHECK-NEXT: %{{.+}} = stablehlo.divide %[[RSS]], %[[SIG2]] : tensor<f64>
  // CHECK-NOT: stablehlo.dot_general %[[XC]]
  // CHECK-NOT: stablehlo.reduce
  func.func @intercept_plus_dot(%rng : tensor<2xui64>, %Xc : tensor<64x3xf64>, %y : tensor<64xf64>, %trace : tensor<1x5xf64>) -> (tensor<1x5xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %Xc, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_Xc: tensor<64x3xf64>, %arg_y: tensor<64xf64>):
      %int:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lx: tensor<1xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
      %b:2 = impulse.sample_region(%int#0, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<3xf64>
      } logpdf {
      ^bb0(%lx: tensor<3xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<2>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<3xf64>)
      %sig:2 = impulse.sample_region(%b#0, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lx: tensor<1xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<3>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<POSITIVE>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
      %xb = stablehlo.dot_general %arg_Xc, %b#1, contracting_dims = [1] x [0] : (tensor<64x3xf64>, tensor<3xf64>) -> tensor<64xf64>
      %ib = stablehlo.broadcast_in_dim %int#1, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %mu = stablehlo.add %ib, %xb : tensor<64xf64>
      %sig_s = stablehlo.reshape %sig#1 : (tensor<1xf64>) -> tensor<f64>
      %sig_b = stablehlo.broadcast_in_dim %sig_s, dims = [] : (tensor<f64>) -> tensor<64xf64>
      %cen = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %z = stablehlo.divide %cen, %sig_b : tensor<64xf64>
      %z2 = stablehlo.multiply %z, %z : tensor<64xf64>
      %log2pi = stablehlo.constant dense<1.8378770664093453> : tensor<64xf64>
      %sm = stablehlo.add %z2, %log2pi : tensor<64xf64>
      %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %ss = stablehlo.reduce(%sm init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<64xf64>, tensor<f64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<5xf64>
      impulse.yield %sig#0, %ss_b : tensor<2xui64>, tensor<5xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]]}> : (tensor<2xui64>, tensor<64x3xf64>, tensor<64xf64>, tensor<1x5xf64>, tensor<f64>) -> (tensor<1x5xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x5xf64>, tensor<1x5xf64>, tensor<f64>, tensor<f64>, tensor<1x5xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x5xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @dot_plus_intercept_transposed
  // CHECK-DAG: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<64x1xf64>
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[XCT:.+]]: tensor<3x64xf64>, %[[Y:.+]]: tensor<64xf64>):
  // CHECK: %[[INT:.+]]:2 = impulse.sample_region
  // CHECK: %[[B:.+]]:2 = impulse.sample_region
  // CHECK: %[[XC:.+]] = stablehlo.transpose %[[XCT]], dims = [1, 0] : (tensor<3x64xf64>) -> tensor<64x3xf64>
  // CHECK-NEXT: %[[X:.+]] = stablehlo.concatenate %[[XC]], %[[ONES]], dim = 1 : (tensor<64x3xf64>, tensor<64x1xf64>) -> tensor<64x4xf64>
  // CHECK-NEXT: %[[BETA:.+]] = stablehlo.concatenate %[[B]]#1, %[[INT]]#1, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
  // CHECK-NEXT: %[[YCOL:.+]] = stablehlo.reshape %[[Y]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[AUG:.+]] = stablehlo.concatenate %[[X]], %[[YCOL]], dim = 1 : (tensor<64x4xf64>, tensor<64x1xf64>) -> tensor<64x5xf64>
  // CHECK-NEXT: %{{.+}}, %{{.+}}, %{{.+}} = enzymexla.linalg.qr %[[AUG]] : (tensor<64x5xf64>) -> (tensor<64x5xf64>, tensor<64x5xf64>, tensor<i32>)
  // CHECK: %{{.+}} = stablehlo.dot_general %{{.+}}, %[[BETA]], contracting_dims = [1] x [0] : (tensor<4x4xf64>, tensor<4xf64>) -> tensor<4xf64>
  // CHECK-NOT: stablehlo.dot_general %[[B]]#1, %[[XCT]]
  func.func @dot_plus_intercept_transposed(%rng : tensor<2xui64>, %XcT : tensor<3x64xf64>, %y : tensor<64xf64>, %trace : tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %XcT, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_XcT: tensor<3x64xf64>, %arg_y: tensor<64xf64>):
      %int:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lx: tensor<1xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
      %b:2 = impulse.sample_region(%int#0, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<3xf64>
      } logpdf {
      ^bb0(%lx: tensor<3xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<2>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<3xf64>)
      %xb = stablehlo.dot_general %b#1, %arg_XcT, contracting_dims = [0] x [0] : (tensor<3xf64>, tensor<3x64xf64>) -> tensor<64xf64>
      %ib = stablehlo.broadcast_in_dim %int#1, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %mu = stablehlo.add %xb, %ib : tensor<64xf64>
      %r = stablehlo.subtract %mu, %arg_y : tensor<64xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<4xf64>
      impulse.yield %b#0, %ss_b : tensor<2xui64>, tensor<4xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]}> : (tensor<2xui64>, tensor<3x64xf64>, tensor<64xf64>, tensor<1x4xf64>, tensor<f64>) -> (tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x4xf64>, tensor<1x4xf64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @intercept_minus_dot
  // CHECK-DAG: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<64x1xf64>
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[XC:.+]]: tensor<64x3xf64>, %[[Y:.+]]: tensor<64xf64>):
  // CHECK: %[[INT:.+]]:2 = impulse.sample_region
  // CHECK: %[[B:.+]]:2 = impulse.sample_region
  // CHECK: %[[NXC:.+]] = stablehlo.negate %[[XC]] : tensor<64x3xf64>
  // CHECK-NEXT: %[[X:.+]] = stablehlo.concatenate %[[ONES]], %[[NXC]], dim = 1 : (tensor<64x1xf64>, tensor<64x3xf64>) -> tensor<64x4xf64>
  // CHECK-NEXT: %[[BETA:.+]] = stablehlo.concatenate %[[INT]]#1, %[[B]]#1, dim = 0 : (tensor<1xf64>, tensor<3xf64>) -> tensor<4xf64>
  // CHECK-NEXT: %[[YCOL:.+]] = stablehlo.reshape %[[Y]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[AUG:.+]] = stablehlo.concatenate %[[X]], %[[YCOL]], dim = 1 : (tensor<64x4xf64>, tensor<64x1xf64>) -> tensor<64x5xf64>
  // CHECK-NEXT: %{{.+}}, %{{.+}}, %{{.+}} = enzymexla.linalg.qr %[[AUG]] : (tensor<64x5xf64>) -> (tensor<64x5xf64>, tensor<64x5xf64>, tensor<i32>)
  func.func @intercept_minus_dot(%rng : tensor<2xui64>, %Xc : tensor<64x3xf64>, %y : tensor<64xf64>, %trace : tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %Xc, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_Xc: tensor<64x3xf64>, %arg_y: tensor<64xf64>):
      %int:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lx: tensor<1xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
      %b:2 = impulse.sample_region(%int#0, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<3xf64>
      } logpdf {
      ^bb0(%lx: tensor<3xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<2>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<3xf64>)
      %xb = stablehlo.dot_general %arg_Xc, %b#1, contracting_dims = [1] x [0] : (tensor<64x3xf64>, tensor<3xf64>) -> tensor<64xf64>
      %ib = stablehlo.broadcast_in_dim %int#1, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %mu = stablehlo.subtract %ib, %xb : tensor<64xf64>
      %r = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<4xf64>
      impulse.yield %b#0, %ss_b : tensor<2xui64>, tensor<4xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]}> : (tensor<2xui64>, tensor<64x3xf64>, tensor<64xf64>, tensor<1x4xf64>, tensor<f64>) -> (tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x4xf64>, tensor<1x4xf64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @folded_dot
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[XA:.+]]: tensor<64x4xf64>, %[[Y:.+]]: tensor<64xf64>):
  // CHECK: %[[INT:.+]]:2 = impulse.sample_region
  // CHECK: %[[B:.+]]:2 = impulse.sample_region
  // CHECK: %[[BETA:.+]] = stablehlo.concatenate %[[INT]]#1, %[[B]]#1, dim = 0 : (tensor<1xf64>, tensor<3xf64>) -> tensor<4xf64>
  // CHECK-NEXT: %[[YCOL:.+]] = stablehlo.reshape %[[Y]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[AUG:.+]] = stablehlo.concatenate %[[XA]], %[[YCOL]], dim = 1 : (tensor<64x4xf64>, tensor<64x1xf64>) -> tensor<64x5xf64>
  // CHECK-NEXT: %{{.+}}, %{{.+}}, %{{.+}} = enzymexla.linalg.qr %[[AUG]] : (tensor<64x5xf64>) -> (tensor<64x5xf64>, tensor<64x5xf64>, tensor<i32>)
  // CHECK: %{{.+}} = stablehlo.dot_general %{{.+}}, %[[BETA]], contracting_dims = [1] x [0] : (tensor<4x4xf64>, tensor<4xf64>) -> tensor<4xf64>
  // CHECK-NOT: stablehlo.dot_general %[[XA]], %[[BETA]]
  func.func @folded_dot(%rng : tensor<2xui64>, %Xa : tensor<64x4xf64>, %y : tensor<64xf64>, %trace : tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %Xa, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_Xa: tensor<64x4xf64>, %arg_y: tensor<64xf64>):
      %int:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lx: tensor<1xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
      %b:2 = impulse.sample_region(%int#0, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<3xf64>
      } logpdf {
      ^bb0(%lx: tensor<3xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<2>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<3xf64>)
      %beta = stablehlo.concatenate %int#1, %b#1, dim = 0 : (tensor<1xf64>, tensor<3xf64>) -> tensor<4xf64>
      %mu = stablehlo.dot_general %arg_Xa, %beta, contracting_dims = [1] x [0] : (tensor<64x4xf64>, tensor<4xf64>) -> tensor<64xf64>
      %r = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<4xf64>
      impulse.yield %b#0, %ss_b : tensor<2xui64>, tensor<4xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]}> : (tensor<2xui64>, tensor<64x4xf64>, tensor<64xf64>, tensor<1x4xf64>, tensor<f64>) -> (tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x4xf64>, tensor<1x4xf64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @neg_design_dependent
  // CHECK-NOT: enzymexla.linalg.qr
  // CHECK: %[[XB:.+]] = stablehlo.dot_general %{{.+}}, %{{.+}}, contracting_dims = [1] x [0] : (tensor<64x3xf64>, tensor<3xf64>) -> tensor<64xf64>
  // CHECK: stablehlo.dot_general %[[R:.+]], %[[R]], contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
  func.func @neg_design_dependent(%rng : tensor<2xui64>, %Xc : tensor<64x3xf64>, %y : tensor<64xf64>, %trace : tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %Xc, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_Xc: tensor<64x3xf64>, %arg_y: tensor<64xf64>):
      %int:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<1xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lx: tensor<1xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
      %b:2 = impulse.sample_region(%int#0, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<3xf64>
      } logpdf {
      ^bb0(%lx: tensor<3xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<2>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<3xf64>)
      %scale = stablehlo.broadcast_in_dim %int#1, dims = [0] : (tensor<1xf64>) -> tensor<64x3xf64>
      %Xdep = stablehlo.multiply %scale, %arg_Xc : tensor<64x3xf64>
      %xb = stablehlo.dot_general %Xdep, %b#1, contracting_dims = [1] x [0] : (tensor<64x3xf64>, tensor<3xf64>) -> tensor<64xf64>
      %ib = stablehlo.broadcast_in_dim %int#1, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
      %mu = stablehlo.add %ib, %xb : tensor<64xf64>
      %r = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<4xf64>
      impulse.yield %b#0, %ss_b : tensor<2xui64>, tensor<4xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]}> : (tensor<2xui64>, tensor<64x3xf64>, tensor<64xf64>, tensor<1x4xf64>, tensor<f64>) -> (tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x4xf64>, tensor<1x4xf64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @neg_intercept_vector
  // CHECK-NOT: enzymexla.linalg.qr
  // CHECK: %[[XB:.+]] = stablehlo.dot_general %{{.+}}, %{{.+}}, contracting_dims = [1] x [0] : (tensor<64x3xf64>, tensor<3xf64>) -> tensor<64xf64>
  // CHECK-NEXT: %[[MU:.+]] = stablehlo.add %{{.+}}, %[[XB]] : tensor<64xf64>
  // CHECK-NEXT: %[[R:.+]] = stablehlo.subtract %{{.+}}, %[[MU]] : tensor<64xf64>
  // CHECK-NEXT: stablehlo.dot_general %[[R]], %[[R]], contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
  // CHECK-NOT: enzymexla.linalg.qr
  func.func @neg_intercept_vector(%rng : tensor<2xui64>, %Xc : tensor<64x3xf64>, %y : tensor<64xf64>, %trace : tensor<1x67xf64>) -> (tensor<1x67xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %Xc, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_Xc: tensor<64x3xf64>, %arg_y: tensor<64xf64>):
      %alpha:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<64xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<64xf64>
      } logpdf {
      ^bb0(%lx: tensor<64xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<64xf64>)
      %b:2 = impulse.sample_region(%alpha#0, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<3xf64>
      } logpdf {
      ^bb0(%lx: tensor<3xf64>, %s: tensor<f64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<2>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<3xf64>)
      %xb = stablehlo.dot_general %arg_Xc, %b#1, contracting_dims = [1] x [0] : (tensor<64x3xf64>, tensor<3xf64>) -> tensor<64xf64>
      %mu = stablehlo.add %alpha#1, %xb : tensor<64xf64>
      %r = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %ss = stablehlo.dot_general %r, %r, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<67xf64>
      impulse.yield %b#0, %ss_b : tensor<2xui64>, tensor<67xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]}> : (tensor<2xui64>, tensor<64x3xf64>, tensor<64xf64>, tensor<1x67xf64>, tensor<f64>) -> (tensor<1x67xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x67xf64>, tensor<1x67xf64>, tensor<f64>, tensor<f64>, tensor<1x67xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x67xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
