// RUN: enzymexlamlir-opt %s --sicm | FileCheck %s
// RUN: enzymexlamlir-opt %s --sicm --lower-enzymexla-linalg="backend=cpu blas_int_width=64" | FileCheck %s --check-prefix=LOWER

module {
  func.func private @normal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %std : tensor<2xui64>, tensor<f64>
  }
  func.func private @normal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  // CHECK-LABEL: func.func @linreg_aug
  // CHECK-SAME: (%{{.+}}: tensor<2xui64>, %[[X:.+]]: tensor<64x2xf64>, %[[Y:.+]]: tensor<64xf64>, %{{.+}}: tensor<1x1xf64>)
  // CHECK: %[[YCOL:.+]] = stablehlo.reshape %[[Y]] : (tensor<64xf64>) -> tensor<64x1xf64>
  // CHECK-NEXT: %[[AUG:.+]] = stablehlo.concatenate %[[X]], %[[YCOL]], dim = 1 : (tensor<64x2xf64>, tensor<64x1xf64>) -> tensor<64x3xf64>
  // CHECK-NEXT: %[[Q:.+]], %[[P:.+]], %{{.+}} = enzymexla.linalg.qr %[[AUG]] : (tensor<64x3xf64>) -> (tensor<64x3xf64>, tensor<64x3xf64>, tensor<i32>)
  // R is the upper triangle of P's leading 2x2 block (a 0/1 mask multiply).
  // CHECK-NEXT: %[[RPACK:.+]] = stablehlo.slice %[[P]] [0:2, 0:2] : (tensor<64x3xf64>) -> tensor<2x2xf64>
  // CHECK-NEXT: %[[ROWS:.+]] = stablehlo.iota dim = 0 : tensor<2x2xi64>
  // CHECK-NEXT: %[[COLS:.+]] = stablehlo.iota dim = 1 : tensor<2x2xi64>
  // CHECK-NEXT: %[[UPPER:.+]] = stablehlo.compare LE, %[[ROWS]], %[[COLS]] : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi1>
  // CHECK-NEXT: %[[MASK:.+]] = stablehlo.convert %[[UPPER]] : (tensor<2x2xi1>) -> tensor<2x2xf64>
  // CHECK-NEXT: %[[R:.+]] = stablehlo.multiply %[[RPACK]], %[[MASK]] : tensor<2x2xf64>
  // CHECK-NEXT: %[[CCOL:.+]] = stablehlo.slice %[[P]] [0:2, 2:3] : (tensor<64x3xf64>) -> tensor<2x1xf64>
  // CHECK-NEXT: %[[C:.+]] = stablehlo.reshape %[[CCOL]] : (tensor<2x1xf64>) -> tensor<2xf64>
  // CHECK-NEXT: %[[PDD:.+]] = stablehlo.slice %[[P]] [2:3, 2:3] : (tensor<64x3xf64>) -> tensor<1x1xf64>
  // CHECK-NEXT: %[[PD:.+]] = stablehlo.reshape %[[PDD]] : (tensor<1x1xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[REM:.+]] = stablehlo.multiply %[[PD]], %[[PD]] : tensor<f64>
  // CHECK-NEXT: impulse.infer_region
  // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
  // CHECK: %[[BETA:.+]] = stablehlo.broadcast_in_dim %[[TAU]]#1, dims = [] : (tensor<f64>) -> tensor<2xf64>
  // CHECK-NEXT: %[[RB:.+]] = stablehlo.dot_general %[[R]], %[[BETA]], contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
  // CHECK-NEXT: %[[D:.+]] = stablehlo.subtract %[[C]], %[[RB]] : tensor<2xf64>
  // CHECK-NEXT: %[[DD:.+]] = stablehlo.dot_general %[[D]], %[[D]], contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  // CHECK-NEXT: %{{.+}} = stablehlo.add %[[DD]], %[[REM]] : tensor<f64>
  // CHECK-NOT: %[[Q]]
  // CHECK-NOT: tensor<64xf64>) -> tensor<f64>

  // LOWER-LABEL: func.func @linreg_aug
  // LOWER: %[[AUG:.+]] = stablehlo.concatenate %{{.+}}, %{{.+}}, dim = 1 : (tensor<64x2xf64>, tensor<64x1xf64>) -> tensor<64x3xf64>
  // LOWER-NEXT: %[[P:.+]], %{{.+}}, %{{.+}} = enzymexla.lapack.geqrf %[[AUG]] : (tensor<64x3xf64>) -> (tensor<64x3xf64>, tensor<64x3xf64>, tensor<i32>)
  // LOWER-NEXT: %{{.+}} = stablehlo.slice %[[P]] [0:2, 0:2] : (tensor<64x3xf64>) -> tensor<2x2xf64>
  // LOWER-NOT: enzymexla.lapack.orgqr
  func.func @linreg_aug(%rng : tensor<2xui64>, %X : tensor<64x2xf64>, %y : tensor<64xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %X, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_X: tensor<64x2xf64>, %arg_y: tensor<64xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        impulse.yield %r, %s : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%lx: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %lx : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %beta = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<2xf64>
      %mu = stablehlo.dot_general %arg_X, %beta, contracting_dims = [1] x [0] : (tensor<64x2xf64>, tensor<2xf64>) -> tensor<64xf64>
      %cen = stablehlo.subtract %arg_y, %mu : tensor<64xf64>
      %ss = stablehlo.dot_general %cen, %cen, contracting_dims = [0] x [0] : (tensor<64xf64>, tensor<64xf64>) -> tensor<f64>
      %neghalf = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
      %lp = stablehlo.multiply %ss, %neghalf : tensor<f64>
      %lp_b = stablehlo.broadcast_in_dim %lp, dims = [] : (tensor<f64>) -> tensor<2xf64>
      impulse.yield %tau#0, %lp_b : tensor<2xui64>, tensor<2xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<64x2xf64>, tensor<64xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
