// RUN: enzymexlamlir-opt %s --inline-infer-regions --canonicalize --cse --sicm --canonicalize --cse --expand-impulse | FileCheck %s

module {
  func.func private @mvn_draw(%rng : tensor<2xui64>, %L : tensor<3x3xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %z = stablehlo.constant dense<1.0> : tensor<3x1xf64>
    %y2 = stablehlo.dot_general %L, %z, contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
    %y = stablehlo.reshape %y2 : (tensor<3x1xf64>) -> tensor<3xf64>
    return %rng, %y : tensor<2xui64>, tensor<3xf64>
  }
  func.func private @mvn_logpdf(%x : tensor<3xf64>, %L : tensor<3x3xf64>)
      -> tensor<f64> {
    %r = stablehlo.reshape %x : (tensor<3xf64>) -> tensor<3x1xf64>
    %s = stablehlo.slice %r [0:1, 0:1] : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %res = stablehlo.reshape %s : (tensor<1x1xf64>) -> tensor<f64>
    return %res : tensor<f64>
  }

  func.func @model(%rng : tensor<2xui64>, %R_data : tensor<3x3xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %L = stablehlo.cholesky %R_data, lower = true : tensor<3x3xf64>
    %y:2 = impulse.sample @mvn_draw(%rng, %L) {
      logpdf = @mvn_logpdf, symbol = #impulse.symbol<1>
    } : (tensor<2xui64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  // CHECK-LABEL: func.func @mh
  // CHECK-SAME: (%[[RNG:.+]]: tensor<2xui64>, %[[R:.+]]: tensor<3x3xf64>, %[[TR:.+]]: tensor<1x3xf64>, %[[WT:.+]]: tensor<f64>)
  // CHECK-DAG: %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
  // CHECK-DAG: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
  // CHECK-DAG: %[[FALSE:.+]] = arith.constant dense<false> : tensor<i1>
  // CHECK-DAG: %[[C1:.+]] = arith.constant dense<1> : tensor<i64>
  // CHECK-DAG: %[[C4:.+]] = arith.constant dense<4> : tensor<i64>
  // CHECK-DAG: %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
  // CHECK-DAG: %[[Z:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3x1xf64>
  // CHECK: %[[CHOL:.+]] = stablehlo.cholesky %[[R]], lower = true : tensor<3x3xf64>
  // CHECK: %[[LOOP:.+]]:4 = impulse.for(%[[C0]] : tensor<i64>) to(%[[C4]] : tensor<i64>) step(%[[C1]] : tensor<i64>) iter_args(%[[TR]], %[[WT]], %[[RNG]], %[[FALSE]] : tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<i1>)
  // CHECK-NEXT: ^bb0(%[[IV:.+]]: tensor<i64>, %[[TRA:.+]]: tensor<1x3xf64>, %[[WTA:.+]]: tensor<f64>, %[[RNGA:.+]]: tensor<2xui64>, %[[ACCA:.+]]: tensor<i1>):
  // CHECK-NEXT: %[[PROP:.+]] = stablehlo.dot_general %[[CHOL]], %[[Z]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[PROPV:.+]] = stablehlo.reshape %[[PROP]] : (tensor<3x1xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[PFLAT:.+]] = impulse.reshape %[[PROPV]] : (tensor<3xf64>) -> tensor<1x3xf64>
  // CHECK-NEXT: %[[NEWTR:.+]] = impulse.dynamic_update_slice %[[TRA]], %[[PFLAT]], %[[C0]], %[[C0]] : (tensor<1x3xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
  // CHECK-NEXT: %[[LPR:.+]] = stablehlo.reshape %[[PROPV]] : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[LPS:.+]] = stablehlo.slice %[[LPR]] [0:1, 0:1] : (tensor<3x1xf64>) -> tensor<1x1xf64>
  // CHECK-NEXT: %[[NEWWT:.+]] = stablehlo.reshape %[[LPS]] : (tensor<1x1xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[OSL:.+]] = impulse.slice %[[TRA]] {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x3xf64>) -> tensor<1x3xf64>
  // CHECK-NEXT: %[[OV:.+]] = impulse.reshape %[[OSL]] : (tensor<1x3xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[OR:.+]] = stablehlo.reshape %[[OV]] : (tensor<3xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[OS:.+]] = stablehlo.slice %[[OR]] [0:1, 0:1] : (tensor<3x1xf64>) -> tensor<1x1xf64>
  // CHECK-NEXT: %[[OLDWT:.+]] = stablehlo.reshape %[[OS]] : (tensor<1x1xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[CORR:.+]] = arith.subf %[[NEWWT]], %[[OLDWT]] : tensor<f64>
  // CHECK-NEXT: %[[WDIFF:.+]] = arith.subf %[[NEWWT]], %[[WTA]] : tensor<f64>
  // CHECK-NEXT: %[[ALPHA:.+]] = arith.subf %[[WDIFF]], %[[CORR]] : tensor<f64>
  // CHECK-NEXT: %[[NEWRNG:.+]], %[[U:.+]] = impulse.random %[[RNGA]], %[[ZERO]], %[[ONE]] {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  // CHECK-NEXT: %[[LOGU:.+]] = math.log %[[U]] : tensor<f64>
  // CHECK-NEXT: %[[ACC:.+]] = arith.cmpf olt, %[[LOGU]], %[[ALPHA]] : tensor<f64>
  // CHECK-NEXT: %[[NEWW:.+]] = impulse.slice %[[NEWTR]] {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x3xf64>) -> tensor<1x3xf64>
  // CHECK-NEXT: %[[OLDW:.+]] = impulse.slice %[[TRA]] {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x3xf64>) -> tensor<1x3xf64>
  // CHECK-NEXT: %[[SELW:.+]] = impulse.select %[[ACC]], %[[NEWW]], %[[OLDW]] : (tensor<i1>, tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<1x3xf64>
  // CHECK-NEXT: %[[SELTR:.+]] = impulse.dynamic_update_slice %[[TRA]], %[[SELW]], %[[C0]], %[[C0]] : (tensor<1x3xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
  // CHECK-NEXT: %[[SELWT:.+]] = arith.select %[[ACC]], %[[NEWWT]], %[[WTA]] : tensor<i1>, tensor<f64>
  // CHECK-NEXT: impulse.yield %[[SELTR]], %[[SELWT]], %[[NEWRNG]], %[[ACC]] : tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<i1>
  // CHECK: return %[[LOOP]]#0, %[[LOOP]]#3, %[[LOOP]]#2, %[[LOOP]]#1 : tensor<1x3xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>
  func.func @mh(%rng : tensor<2xui64>, %R_data : tensor<3x3xf64>, %trace : tensor<1x3xf64>, %weight : tensor<f64>)
      -> (tensor<1x3xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>) {
    %res:5 = impulse.infer @model(%rng, %R_data) given %trace state = [%weight] {
      mh_config = #impulse.mh_config,
      selection = [[#impulse.symbol<1>]],
      all_addresses = [[#impulse.symbol<1>]],
      state_schema = ["weight"],
      num_samples = 4 : i64
    } : (tensor<2xui64>, tensor<3x3xf64>, tensor<f64>, tensor<1x3xf64>)
        -> (tensor<1x3xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1, %res#3, %res#4 : tensor<1x3xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>
  }
}
