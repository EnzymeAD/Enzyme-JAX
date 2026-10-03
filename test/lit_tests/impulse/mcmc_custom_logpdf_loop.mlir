// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(expand-impulse,lower-impulse-to-stablehlo{backend=cpu},outline-enzyme-regions,enzyme,canonicalize,remove-unnecessary-enzyme-ops,canonicalize,cse)" | FileCheck %s

// Logpdf cloned, not called: one forward and one reverse loop per gradient.

module {
  // CHECK-LABEL: func.func private @diffehmc_loop_logpdf_to_diff0(
  // CHECK-SAME: %[[Q0:.+]]: tensor<2xf64>, %[[RNG0:.+]]: tensor<2xui64>, %{{.+}}: tensor<f64>)
  // CHECK-NOT: call
  // CHECK: %[[FWD0:.+]]:4 = stablehlo.while(%{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}, %{{.+}} = %[[Q0]], %{{.+}} = %{{.+}}) : tensor<i64>, tensor<f64>, tensor<2xf64>, tensor<4x2xf64>
  // CHECK: stablehlo.dynamic_update_slice
  // CHECK: stablehlo.sine
  // CHECK: %[[LP0:.+]] = stablehlo.multiply %{{.+}}, %[[FWD0]]#1 : tensor<f64>
  // CHECK-NEXT: %[[U0:.+]] = arith.negf %[[LP0]] : tensor<f64>
  // CHECK-NOT: call
  // CHECK: stablehlo.while
  // CHECK: stablehlo.dynamic_slice %[[FWD0]]#3
  // CHECK: stablehlo.cosine
  // CHECK-NOT: stablehlo.while
  // CHECK-NOT: call
  // CHECK: return %[[U0]], %[[RNG0]], %{{.+}} : tensor<f64>, tensor<2xui64>, tensor<2xf64>
  // CHECK-LABEL: func.func private @diffehmc_loop_logpdf_to_diff1(
  // CHECK-SAME: %[[Q1:.+]]: tensor<2xf64>, %[[RNG1:.+]]: tensor<2xui64>, %{{.+}}: tensor<f64>)
  // CHECK-NOT: call
  // CHECK: %[[FWD1:.+]]:4 = stablehlo.while(%{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}, %{{.+}} = %[[Q1]], %{{.+}} = %{{.+}}) : tensor<i64>, tensor<f64>, tensor<2xf64>, tensor<4x2xf64>
  // CHECK: stablehlo.dynamic_update_slice
  // CHECK: stablehlo.sine
  // CHECK: %[[LP1:.+]] = stablehlo.multiply %{{.+}}, %[[FWD1]]#1 : tensor<f64>
  // CHECK-NEXT: %[[U1:.+]] = arith.negf %[[LP1]] : tensor<f64>
  // CHECK-NOT: call
  // CHECK: stablehlo.while
  // CHECK: stablehlo.dynamic_slice %[[FWD1]]#3
  // CHECK: stablehlo.cosine
  // CHECK-NOT: stablehlo.while
  // CHECK-NOT: call
  // CHECK: return %[[U1]], %[[RNG1]], %{{.+}} : tensor<f64>, tensor<2xui64>, tensor<2xf64>

  func.func @loop_logpdf(%x : tensor<2xf64>) -> tensor<f64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %n = stablehlo.constant dense<4> : tensor<i64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %r:3 = stablehlo.while(%i = %c0, %acc = %zero, %v = %x) : tensor<i64>, tensor<f64>, tensor<2xf64>
     cond {
      %lt = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %lt : tensor<i1>
    } do {
      %sq = stablehlo.dot_general %v, %v, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
      %acc2 = stablehlo.add %acc, %sq : tensor<f64>
      %v2 = stablehlo.sine %v : tensor<2xf64>
      %i2 = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %i2, %acc2, %v2 : tensor<i64>, tensor<f64>, tensor<2xf64>
    }
    %neg_half = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
    %res = stablehlo.multiply %neg_half, %r#1 : tensor<f64>
    return %res : tensor<f64>
  }

  func.func @hmc_loop_logpdf(%rng : tensor<2xui64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_pos = stablehlo.constant dense<[[0.5, -0.5]]> : tensor<1x2xf64>
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %res:4 = "impulse.infer"(%rng, %step_size, %init_pos) {
      logpdf_fn = @loop_logpdf,
      hmc_config = #impulse.hmc_config<trajectory_length = 1.000000e+00 : f64, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "hmc_loop_logpdf",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 1, 0, 0, 0, 0, 1, 1, 0>
    } : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}
