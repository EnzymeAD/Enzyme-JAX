// RUN: enzymexlamlir-opt %s --inline-infer-regions --sicm | FileCheck %s

// The slice of mu hoists above the rebuilt if; arm-local ops stay in the arm.

module {
  func.func private @draw(%rng : tensor<2xui64>, %m : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %m : tensor<2xui64>, tensor<f64>
  }
  func.func private @draw1(%rng : tensor<2xui64>, %m : tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>) {
    return %rng, %m : tensor<2xui64>, tensor<1xf64>
  }
  func.func private @lp_a(%x : tensor<f64>, %m : tensor<f64>) -> tensor<f64> {
    %d = stablehlo.subtract %x, %m : tensor<f64>
    %n = stablehlo.multiply %d, %d : tensor<f64>
    return %n : tensor<f64>
  }
  func.func private @lp_out(%y : tensor<f64>, %m : tensor<f64>) -> tensor<f64> {
    %c = stablehlo.constant dense<-1.3862943611198906> : tensor<f64>
    return %c : tensor<f64>
  }
  func.func private @lp_in(%y : tensor<1xf64>, %m : tensor<1xf64>) -> tensor<f64> {
    %d = stablehlo.subtract %y, %m : tensor<1xf64>
    %n = stablehlo.multiply %d, %d : tensor<1xf64>
    %r = stablehlo.reshape %n : (tensor<1xf64>) -> tensor<f64>
    return %r : tensor<f64>
  }

  func.func @model(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %xs : tensor<2xf64>) -> (tensor<2xui64>, tensor<1xf64>) {
    %a:2 = impulse.sample @draw(%rng, %m) {logpdf = @lp_a, symbol = #impulse.symbol<1>, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %ab = stablehlo.broadcast_in_dim %a#1, dims = [] : (tensor<f64>) -> tensor<2xf64>
    %mu = stablehlo.multiply %ab, %xs : tensor<2xf64>
    %z:2 = impulse.sample(%a#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<2>, name = "z"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %half = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %two = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %p = stablehlo.compare GT, %z#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %if:2 = impulse.if (%p) ({
      %yo:2 = impulse.sample @draw(%z#0, %m) {logpdf = @lp_out, symbol = #impulse.symbol<3>, name = "yo"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %yo1 = stablehlo.reshape %yo#1 : (tensor<f64>) -> tensor<1xf64>
      impulse.yield %yo#0, %yo1 : tensor<2xui64>, tensor<1xf64>
    }, {
      %s = stablehlo.slice %mu [0:1] : (tensor<2xf64>) -> tensor<1xf64>
      %r = stablehlo.reshape %s : (tensor<1xf64>) -> tensor<f64>
      %t = stablehlo.multiply %r, %two : tensor<f64>
      %t1 = stablehlo.reshape %t : (tensor<f64>) -> tensor<1xf64>
      %yi:2 = impulse.sample @draw1(%z#0, %t1) {logpdf = @lp_in, symbol = #impulse.symbol<4>, name = "yi"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yi#0, %yi#1 : tensor<2xui64>, tensor<1xf64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<1xf64>)
    return %if#0, %if#1 : tensor<2xui64>, tensor<1xf64>
  }

  func.func @gibbs(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %xs : tensor<2xf64>, %trace : tensor<1x4xf64>, %obs : tensor<1x2xf64>, %w : tensor<f64>) -> tensor<1x4xf64> {
    %res:5 = impulse.infer @model(%rng, %m, %logit, %xs) given %trace constrained_by %obs state = [%w]
        { gibbs_config = #impulse.gibbs_config,
          selection = [[#impulse.symbol<2>]],
          all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
          constrained_addresses = [[#impulse.symbol<3>], [#impulse.symbol<4>]],
          state_schema = ["weight"], num_samples = 1 : i64 }
        : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<1x4xf64>, tensor<1x2xf64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0 : tensor<1x4xf64>
  }
}

// CHECK-LABEL: func.func @gibbs
// CHECK: %[[MU:.+]] = stablehlo.multiply %{{.+}}, %{{.+}} : tensor<2xf64>
// CHECK: %[[S:.+]] = stablehlo.slice %[[MU]] [0:1] : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[R:.+]] = stablehlo.reshape %[[S]] : (tensor<1xf64>) -> tensor<f64>
// CHECK: ^bb0(%[[ZV:[^:]+]]: tensor<f64>):
// CHECK: %[[IF:.+]]:5 = impulse.if(%{{.+}}) ({
// CHECK-NEXT: impulse.yield
// CHECK-NEXT: }, {
// CHECK-NEXT: %[[T:.+]] = stablehlo.multiply %[[R]], %{{.+}} : tensor<f64>
// CHECK-NEXT: %[[T1:.+]] = stablehlo.reshape %[[T]] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[D:.+]] = stablehlo.subtract %{{.+}}, %[[T1]] : tensor<1xf64>
// CHECK-NEXT: %[[SQ:.+]] = stablehlo.multiply %[[D]], %[[D]] : tensor<1xf64>
// CHECK-NEXT: %[[LIN:.+]] = stablehlo.reshape %[[SQ]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: impulse.yield %{{.+}}, %{{.+}}, %{{.+}}, %[[LIN]], %[[LIN]]
// CHECK-NEXT: }) : (tensor<i1>) -> (tensor<2xui64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>)
