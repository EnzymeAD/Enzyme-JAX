// RUN: enzymexlamlir-opt %s --inline-infer-regions --canonicalize --cse --sicm --canonicalize --cse | FileCheck %s --check-prefix=REGION
// RUN: enzymexlamlir-opt %s --inline-infer-regions --canonicalize --cse --sicm --canonicalize --cse --expand-impulse --canonicalize --cse | FileCheck %s --check-prefix=GIBBS

module {
  func.func private @draw(%rng : tensor<2xui64>, %m : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %m : tensor<2xui64>, tensor<f64>
  }
  func.func private @lp_a(%x : tensor<f64>, %m : tensor<f64>) -> tensor<f64> {
    %d = stablehlo.subtract %x, %m : tensor<f64>
    %n = stablehlo.multiply %d, %d : tensor<f64>
    %r = stablehlo.negate %n : tensor<f64>
    return %r : tensor<f64>
  }
  func.func private @lp_in(%y : tensor<f64>, %m : tensor<f64>) -> tensor<f64> {
    %d = stablehlo.subtract %y, %m : tensor<f64>
    %n = stablehlo.multiply %d, %d : tensor<f64>
    %c = stablehlo.constant dense<-5.000000e+01> : tensor<f64>
    %r = stablehlo.multiply %n, %c : tensor<f64>
    return %r : tensor<f64>
  }
  func.func private @lp_out(%y : tensor<f64>, %m : tensor<f64>) -> tensor<f64> {
    %c = stablehlo.constant dense<-1.3862943611198906> : tensor<f64>
    return %c : tensor<f64>
  }

  func.func @model(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %a:2 = impulse.sample @draw(%rng, %m) <{logpdf = @lp_a, symbol = #impulse.symbol<1>, name = "a"}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %z:2 = impulse.sample(%a#0, %logit) <{dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<2>, name = "z"}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %half = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %p = stablehlo.compare GT, %z#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %if:2 = impulse.if (%p) ({
      %yo:2 = impulse.sample @draw(%z#0, %m) <{logpdf = @lp_out, symbol = #impulse.symbol<3>, name = "yo"}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %yo#0, %yo#1 : tensor<2xui64>, tensor<f64>
    }, {
      %yi:2 = impulse.sample @draw(%z#0, %a#1) <{logpdf = @lp_in, symbol = #impulse.symbol<4>, name = "yi"}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %yi#0, %yi#1 : tensor<2xui64>, tensor<f64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>)
    return %if#0, %if#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @gibbs(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %trace : tensor<1x4xf64>, %obs : tensor<1x2xf64>, %w : tensor<f64>) -> (tensor<1x4xf64>, tensor<2xui64>, tensor<f64>) {
    %res:5 = impulse.infer @model(%rng, %m, %logit) given %trace constrained_by %obs state = [%w]
        <{ gibbs_config = #impulse.gibbs_config,
          selection = [[#impulse.symbol<2>]],
          all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
          constrained_addresses = [[#impulse.symbol<3>], [#impulse.symbol<4>]],
          state_schema = ["weight"],
          num_samples = 1 : i64 }>
        : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>, tensor<1x2xf64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#3, %res#4 : tensor<1x4xf64>, tensor<2xui64>, tensor<f64>
  }
}

// REGION-LABEL: func.func @gibbs
// REGION-SAME: (%[[RNG:[^:]+]]: tensor<2xui64>, %[[M:[^:]+]]: tensor<f64>, %[[LOGIT:[^:]+]]: tensor<f64>, %[[TR:[^:]+]]: tensor<1x4xf64>, %[[OBS:[^:]+]]: tensor<1x2xf64>, %[[W:[^:]+]]: tensor<f64>)
// REGION: %[[ASL:.+]] = impulse.slice %[[TR]] <start_indices = [0, 0], limit_indices = [1, 1], strides = [1, 1]> : (tensor<1x4xf64>) -> tensor<1x1xf64>
// REGION-NEXT: %[[A:.+]] = impulse.reshape %[[ASL]] : (tensor<1x1xf64>) -> tensor<f64>
// REGION-NEXT: %[[YOSL:.+]] = impulse.slice %[[OBS]] <start_indices = [0, 0], limit_indices = [1, 1], strides = [1, 1]> : (tensor<1x2xf64>) -> tensor<1x1xf64>
// REGION-NEXT: %[[YO:.+]] = impulse.reshape %[[YOSL]] : (tensor<1x1xf64>) -> tensor<f64>
// REGION-NEXT: %[[YISL:.+]] = impulse.slice %[[OBS]] <start_indices = [0, 1], limit_indices = [1, 2], strides = [1, 1]> : (tensor<1x2xf64>) -> tensor<1x1xf64>
// REGION-NEXT: %[[YI:.+]] = impulse.reshape %[[YISL]] : (tensor<1x1xf64>) -> tensor<f64>
// REGION: impulse.infer_region @model
// REGION: impulse.if(%{{.+}}) ({
// REGION-NEXT: %{{.+}}:2 = impulse.sample_region
// REGION: } <fn = @draw, logpdf_fn = @lp_out, address = [#impulse.symbol<3>], name = "yo">
// REGION: } <fn = @draw, logpdf_fn = @lp_in, address = [#impulse.symbol<4>], name = "yi">
// REGION: } logpdf {
// REGION-NEXT: ^bb0(%[[ZV:.+]]: tensor<f64>):
// REGION-NEXT: %[[ZL:.+]] = arith.mulf %[[ZV]], %[[LOGIT]] : tensor<f64>
// REGION-NEXT: %[[ZLR:.+]] = impulse.reshape %[[ZL]] : (tensor<f64>) -> tensor<f64>
// REGION-NEXT: %[[LPZ:.+]] = arith.addf %[[ZLR]], %{{.+}} : tensor<f64>
// REGION-NEXT: %[[J1:.+]] = arith.addf %[[LPA:.+]], %[[LPZ]] : tensor<f64>
// REGION-NEXT: %[[P:.+]] = stablehlo.compare GT, %[[ZV]], %{{.+}}, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
// REGION-NEXT: %[[IF:.+]]:5 = impulse.if(%[[P]]) ({
// REGION-NEXT: impulse.yield %[[RNG]], %[[YO]], %[[LOUT:.+]], %[[ZERO:.+]], %[[LOUT]] : tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
// REGION-NEXT: }, {
// REGION-NEXT: %[[D:.+]] = stablehlo.subtract %[[YI]], %[[A]] : tensor<f64>
// REGION-NEXT: %[[SQ:.+]] = stablehlo.multiply %[[D]], %[[D]] : tensor<f64>
// REGION-NEXT: %[[LIN:.+]] = stablehlo.multiply %[[SQ]], %{{.+}} : tensor<f64>
// REGION-NEXT: impulse.yield %[[RNG]], %[[YI]], %[[ZERO]], %[[LIN]], %[[LIN]] : tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
// REGION-NEXT: }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>)
// REGION-NEXT: %[[JOINT:.+]] = arith.addf %[[J1]], %[[IF]]#4 : tensor<f64>
// REGION-NEXT: impulse.yield %[[JOINT]], %[[LPA]], %[[LPZ]], %[[IF]]#2, %[[IF]]#3 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
// REGION-NEXT: } {{.*}}gibbs_config = #impulse.gibbs_config, {{.*}}num_position_args = 1, position_size = 1

// Scores only z's prior and the impulse.if (not a's prior) at z = 0 and z = 1.
// GIBBS-LABEL: func.func @gibbs
// GIBBS-SAME: (%[[RNG:[^:]+]]: tensor<2xui64>, %[[M:[^:]+]]: tensor<f64>, %[[LOGIT:[^:]+]]: tensor<f64>, %[[TR:[^:]+]]: tensor<1x4xf64>, %[[OBS:[^:]+]]: tensor<1x2xf64>, %[[W:[^:]+]]: tensor<f64>)
// GIBBS-DAG: %[[C3:.+]] = arith.constant dense<3> : tensor<i64>
// GIBBS-DAG: %[[C2:.+]] = arith.constant dense<2> : tensor<i64>
// GIBBS-DAG: %[[C1:.+]] = arith.constant dense<1> : tensor<i64>
// GIBBS-DAG: %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// GIBBS-DAG: %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// GIBBS-DAG: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// GIBBS-NOT: impulse.infer
// GIBBS: %[[YOSL:.+]] = impulse.slice %[[OBS]] <start_indices = [0, 0], limit_indices = [1, 1], strides = [1, 1]>
// GIBBS: %[[YISL:.+]] = impulse.slice %[[OBS]] <start_indices = [0, 1], limit_indices = [1, 2], strides = [1, 1]>
// GIBBS: %[[ZSL:.+]] = impulse.slice %[[TR]] <start_indices = [0, 1], limit_indices = [1, 2], strides = [1, 1]> : (tensor<1x4xf64>) -> tensor<1x1xf64>
// GIBBS-NEXT: %[[ZCUR:.+]] = impulse.reshape %[[ZSL]] : (tensor<1x1xf64>) -> tensor<f64>
// GIBBS-NEXT: %[[Z0L:.+]] = arith.mulf %[[LOGIT]], %[[ZERO]] : tensor<f64>
// GIBBS-NEXT: %[[Z0R:.+]] = impulse.reshape %[[Z0L]] : (tensor<f64>) -> tensor<f64>
// GIBBS-NEXT: %[[LPZ0:.+]] = arith.addf %[[Z0R]], %[[SP:.+]] : tensor<f64>
// GIBBS-NEXT: %[[P0:.+]] = stablehlo.compare GT, %[[ZERO]], %{{.+}}, FLOAT
// GIBBS-NEXT: %[[IF0:.+]]:5 = impulse.if(%[[P0]]) ({
// GIBBS: }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>)
// GIBBS-NEXT: %[[S0A:.+]] = arith.addf %[[LPZ0]], %[[IF0]]#2 : tensor<f64>
// GIBBS-NEXT: %[[S0:.+]] = arith.addf %[[S0A]], %[[IF0]]#3 : tensor<f64>
// GIBBS-NEXT: %[[Z1R:.+]] = impulse.reshape %[[LOGIT]] : (tensor<f64>) -> tensor<f64>
// GIBBS-NEXT: %[[LPZ1:.+]] = arith.addf %[[Z1R]], %[[SP]] : tensor<f64>
// GIBBS-NEXT: %[[P1:.+]] = stablehlo.compare GT, %[[ONE]], %{{.+}}, FLOAT
// GIBBS-NEXT: %[[IF1:.+]]:5 = impulse.if(%[[P1]]) ({
// GIBBS: }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>)
// GIBBS-NEXT: %[[S1A:.+]] = arith.addf %[[LPZ1]], %[[IF1]]#2 : tensor<f64>
// GIBBS-NEXT: %[[S1:.+]] = arith.addf %[[S1A]], %[[IF1]]#3 : tensor<f64>
// GIBBS-NEXT: %[[DIFF:.+]] = arith.subf %[[S1]], %[[S0]] : tensor<f64>
// GIBBS-NEXT: %[[PR:.+]] = impulse.logistic %[[DIFF]] : (tensor<f64>) -> tensor<f64>
// GIBBS-NEXT: %[[RNG1:.+]], %[[U:.+]] = impulse.random %[[RNG]], %[[ZERO]], %[[ONE]] <rng_distribution = UNIFORM>
// GIBBS-NEXT: %[[TAKE1:.+]] = arith.cmpf olt, %[[U]], %[[PR]] : tensor<f64>
// GIBBS-NEXT: %[[ZCURS:.+]] = impulse.reshape %[[ZCUR]] : (tensor<f64>) -> tensor<f64>
// GIBBS-NEXT: %[[CUR1:.+]] = arith.cmpf ogt, %[[ZCURS]], %{{.+}} : tensor<f64>
// GIBBS-NEXT: %[[SNEW:.+]] = arith.select %[[TAKE1]], %[[S1]], %[[S0]] : tensor<i1>, tensor<f64>
// GIBBS-NEXT: %[[SCUR:.+]] = arith.select %[[CUR1]], %[[S1]], %[[S0]] : tensor<i1>, tensor<f64>
// GIBBS-NEXT: %[[DW:.+]] = arith.subf %[[SNEW]], %[[SCUR]] : tensor<f64>
// GIBBS-NEXT: %[[WNEW:.+]] = arith.addf %[[W]], %[[DW]] : tensor<f64>
// GIBBS-NEXT: %[[ZNEW:.+]] = arith.select %[[TAKE1]], %[[ONE]], %[[ZERO]] : tensor<i1>, tensor<f64>
// GIBBS-NEXT: %[[ZNEWR:.+]] = impulse.reshape %[[ZNEW]] : (tensor<f64>) -> tensor<1x1xf64>
// GIBBS-NEXT: %[[T1:.+]] = impulse.dynamic_update_slice %[[TR]], %[[ZNEWR]], %[[C0]], %[[C1]] : (tensor<1x4xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x4xf64>
// GIBBS-NEXT: %[[YOR:.+]] = impulse.reshape %[[YOSL]] : (tensor<1x1xf64>) -> tensor<1x1xf64>
// GIBBS-NEXT: %[[T2:.+]] = impulse.dynamic_update_slice %[[T1]], %[[YOR]], %[[C0]], %[[C2]] : (tensor<1x4xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x4xf64>
// GIBBS-NEXT: %[[YIR:.+]] = impulse.reshape %[[YISL]] : (tensor<1x1xf64>) -> tensor<1x1xf64>
// GIBBS-NEXT: %[[T3:.+]] = impulse.dynamic_update_slice %[[T2]], %[[YIR]], %[[C0]], %[[C3]] : (tensor<1x4xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x4xf64>
// GIBBS-NEXT: return %[[T3]], %[[RNG1]], %[[WNEW]]
