// RUN: enzymexlamlir-opt %s --split-input-file --inline-infer-regions --sicm | FileCheck %s
// RUN: enzymexlamlir-opt %s --split-input-file --inline-infer-regions --sicm=max-iterations=0 | FileCheck %s --check-prefix=NOSICM

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
    %h = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
    %r = stablehlo.multiply %h, %n : tensor<f64>
    return %r : tensor<f64>
  }
  func.func private @lp_out(%y : tensor<1xf64>, %m : tensor<1xf64>) -> tensor<f64> {
    %c = stablehlo.constant dense<-1.3862943611198906> : tensor<f64>
    return %c : tensor<f64>
  }
  func.func private @lp_in(%y : tensor<1xf64>, %m : tensor<1xf64>) -> tensor<f64> {
    %d = stablehlo.subtract %y, %m : tensor<1xf64>
    %n = stablehlo.multiply %d, %d : tensor<1xf64>
    %h = stablehlo.constant dense<-5.000000e-01> : tensor<1xf64>
    %q = stablehlo.multiply %h, %n : tensor<1xf64>
    %r = stablehlo.reshape %q : (tensor<1xf64>) -> tensor<f64>
    return %r : tensor<f64>
  }
  func.func @model(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %xs : tensor<3xf64>) -> (tensor<2xui64>, tensor<1xf64>) {
    %a:2 = impulse.sample @draw(%rng, %m) {logpdf = @lp_a, symbol = #impulse.symbol<1>, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %ab = stablehlo.broadcast_in_dim %a#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %mu = stablehlo.multiply %ab, %xs : tensor<3xf64>
    %m1 = stablehlo.constant dense<0.0> : tensor<1xf64>
    %half = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %z0:2 = impulse.sample(%a#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<10>, name = "z0"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p0 = stablehlo.compare GT, %z0#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %if0:2 = impulse.if (%p0) ({
      %yo0:2 = impulse.sample @draw1(%z0#0, %m1) {logpdf = @lp_out, symbol = #impulse.symbol<20>, name = "yo0"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yo0#0, %yo0#1 : tensor<2xui64>, tensor<1xf64>
    }, {
      %s0 = stablehlo.slice %mu [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %yi0:2 = impulse.sample @draw1(%z0#0, %s0) {logpdf = @lp_in, symbol = #impulse.symbol<30>, name = "yi0"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yi0#0, %yi0#1 : tensor<2xui64>, tensor<1xf64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<1xf64>)
    %z1:2 = impulse.sample(%if0#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<11>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p1 = stablehlo.compare GT, %z1#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %if1:2 = impulse.if (%p1) ({
      %yo1:2 = impulse.sample @draw1(%z1#0, %m1) {logpdf = @lp_out, symbol = #impulse.symbol<21>, name = "yo1"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yo1#0, %yo1#1 : tensor<2xui64>, tensor<1xf64>
    }, {
      %s1 = stablehlo.slice %mu [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %yi1:2 = impulse.sample @draw1(%z1#0, %s1) {logpdf = @lp_in, symbol = #impulse.symbol<31>, name = "yi1"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yi1#0, %yi1#1 : tensor<2xui64>, tensor<1xf64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<1xf64>)
    %z2:2 = impulse.sample(%if1#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<12>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p2 = stablehlo.compare GT, %z2#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %if2:2 = impulse.if (%p2) ({
      %yo2:2 = impulse.sample @draw1(%z2#0, %m1) {logpdf = @lp_out, symbol = #impulse.symbol<22>, name = "yo2"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yo2#0, %yo2#1 : tensor<2xui64>, tensor<1xf64>
    }, {
      %s2 = stablehlo.slice %mu [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %yi2:2 = impulse.sample @draw1(%z2#0, %s2) {logpdf = @lp_in, symbol = #impulse.symbol<32>, name = "yi2"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yi2#0, %yi2#1 : tensor<2xui64>, tensor<1xf64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<1xf64>)
    return %if2#0, %if0#1 : tensor<2xui64>, tensor<1xf64>
  }
  func.func @main(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %xs : tensor<3xf64>, %trace : tensor<1x10xf64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step = stablehlo.constant dense<0.1> : tensor<f64>
    %res:4 = impulse.infer @model(%rng, %m, %logit, %xs) given %trace init_step_size = %step {
      selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<10>], [#impulse.symbol<20>], [#impulse.symbol<30>], [#impulse.symbol<11>], [#impulse.symbol<21>], [#impulse.symbol<31>], [#impulse.symbol<12>], [#impulse.symbol<22>], [#impulse.symbol<32>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, num_samples = 1 : i64, thinning = 1 : i64, num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<3xf64>, tensor<1x10xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @main
// CHECK: %[[P0:.+]] = stablehlo.compare GT, %{{.+}}, %{{.+}}, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CHECK-NEXT: %[[P1:.+]] = stablehlo.compare GT, %{{.+}}, %{{.+}}, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CHECK-NEXT: %[[P2:.+]] = stablehlo.compare GT, %{{.+}}, %{{.+}}, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CHECK-NEXT: %[[Q0:.+]] = stablehlo.reshape %[[P0]] : (tensor<i1>) -> tensor<1xi1>
// CHECK-NEXT: %[[Q1:.+]] = stablehlo.reshape %[[P1]] : (tensor<i1>) -> tensor<1xi1>
// CHECK-NEXT: %[[Q2:.+]] = stablehlo.reshape %[[P2]] : (tensor<i1>) -> tensor<1xi1>
// CHECK-NEXT: %[[P:.+]] = stablehlo.concatenate %[[Q0]], %[[Q1]], %[[Q2]], dim = 0 : (tensor<1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<3xi1>
// CHECK-NEXT: %[[Y:.+]] = stablehlo.concatenate %{{.+}}, %{{.+}}, %{{.+}}, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
// CHECK-NEXT: %[[HS:.+]] = stablehlo.reshape %{{.+}} : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[H:.+]] = stablehlo.broadcast_in_dim %[[HS]], dims = [] : (tensor<f64>) -> tensor<3xf64>
// CHECK-NEXT: %[[OUT:.+]] = stablehlo.broadcast_in_dim %{{.+}}, dims = [] : (tensor<f64>) -> tensor<3xf64>
// CHECK-NEXT: %[[ZERO:.+]] = stablehlo.broadcast_in_dim %{{.+}}, dims = [] : (tensor<f64>) -> tensor<3xf64>
// CHECK-NEXT: %{{.+}} = stablehlo.select %[[P]], %[[OUT]], %[[ZERO]] : tensor<3xi1>, tensor<3xf64>

// CHECK: ^bb0(%[[A:arg[0-9]+]]: tensor<f64>):
// CHECK-NOT: impulse.if
// CHECK: %[[AB:.+]] = stablehlo.broadcast_in_dim %[[A]], dims = [] : (tensor<f64>) -> tensor<3xf64>
// CHECK-NEXT: %[[MU:.+]] = stablehlo.multiply %[[AB]], %{{.+}} : tensor<3xf64>
// CHECK-NEXT: %{{.+}} = arith.addf %{{.+}}, %{{.+}} : tensor<f64>
// CHECK-NEXT: %[[D:.+]] = stablehlo.subtract %[[Y]], %[[MU]] : tensor<3xf64>
// CHECK-NEXT: %[[SQ:.+]] = stablehlo.multiply %[[D]], %[[D]] : tensor<3xf64>
// CHECK-NEXT: %[[LIN:.+]] = stablehlo.multiply %[[H]], %[[SQ]] : tensor<3xf64>
// CHECK-NEXT: %[[TIN:.+]] = stablehlo.select %[[P]], %{{.+}}, %[[LIN]] : tensor<3xi1>, tensor<3xf64>
// CHECK-NEXT: %[[TIN0:.+]] = stablehlo.slice %[[TIN]] [0:1] : (tensor<3xf64>) -> tensor<1xf64>
// CHECK-NEXT: %{{.+}} = stablehlo.reshape %[[TIN0]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[TIN1:.+]] = stablehlo.slice %[[TIN]] [1:2] : (tensor<3xf64>) -> tensor<1xf64>
// CHECK-NEXT: %{{.+}} = stablehlo.reshape %[[TIN1]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[TIN2:.+]] = stablehlo.slice %[[TIN]] [2:3] : (tensor<3xf64>) -> tensor<1xf64>
// CHECK-NEXT: %{{.+}} = stablehlo.reshape %[[TIN2]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[TOT:.+]] = stablehlo.select %[[P]], %{{.+}}, %[[LIN]] : tensor<3xi1>, tensor<3xf64>
// CHECK-NEXT: %[[TOT0:.+]] = stablehlo.slice %[[TOT]] [0:1] : (tensor<3xf64>) -> tensor<1xf64>
// CHECK-NOT: impulse.if
// CHECK: hmc_config

// The rewrite is part of logpdf construction, so it also runs without SICM.
// NOSICM-LABEL: func.func @main
// NOSICM: ^bb0(%{{arg[0-9]+}}: tensor<f64>):
// NOSICM-NOT: impulse.if
// NOSICM: %[[MU:.+]] = stablehlo.multiply %{{.+}}, %{{.+}} : tensor<3xf64>
// NOSICM-NOT: impulse.if
// NOSICM: %[[P:.+]] = stablehlo.concatenate %{{.+}}, %{{.+}}, %{{.+}}, dim = 0 : (tensor<1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<3xi1>
// NOSICM-NEXT: %{{.+}} = stablehlo.concatenate %{{.+}}, %{{.+}}, %{{.+}}, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
// NOSICM-NEXT: %[[Y:.+]] = stablehlo.concatenate %{{.+}}, %{{.+}}, %{{.+}}, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
// NOSICM-NEXT: %{{.+}} = arith.constant dense<0.000000e+00> : tensor<f64>
// NOSICM-NEXT: %[[D:.+]] = stablehlo.subtract %[[Y]], %[[MU]] : tensor<3xf64>
// NOSICM-NEXT: %[[SQ:.+]] = stablehlo.multiply %[[D]], %[[D]] : tensor<3xf64>
// NOSICM: %{{.+}} = stablehlo.select %[[P]], %{{.+}}, %{{.+}} : tensor<3xi1>, tensor<3xf64>
// NOSICM-NOT: impulse.if
// NOSICM: hmc_config

// -----

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
    %h = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
    %r = stablehlo.multiply %h, %n : tensor<f64>
    return %r : tensor<f64>
  }
  func.func private @lp_out(%y : tensor<1xf64>, %m : tensor<1xf64>) -> tensor<f64> {
    %c = stablehlo.constant dense<-1.3862943611198906> : tensor<f64>
    return %c : tensor<f64>
  }
  func.func private @lp_in(%y : tensor<1xf64>, %m : tensor<1xf64>) -> tensor<f64> {
    %d = stablehlo.subtract %y, %m : tensor<1xf64>
    %n = stablehlo.multiply %d, %d : tensor<1xf64>
    %h = stablehlo.constant dense<-5.000000e-01> : tensor<1xf64>
    %q = stablehlo.multiply %h, %n : tensor<1xf64>
    %r = stablehlo.reshape %q : (tensor<1xf64>) -> tensor<f64>
    return %r : tensor<f64>
  }
  func.func private @lp_in3(%y : tensor<1xf64>, %m : tensor<1xf64>) -> tensor<f64> {
    %d = stablehlo.subtract %y, %m : tensor<1xf64>
    %n = stablehlo.abs %d : tensor<1xf64>
    %h = stablehlo.constant dense<-5.000000e-01> : tensor<1xf64>
    %q = stablehlo.multiply %h, %n : tensor<1xf64>
    %r = stablehlo.reshape %q : (tensor<1xf64>) -> tensor<f64>
    return %r : tensor<f64>
  }
  func.func @model(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %xs : tensor<2xf64>) -> (tensor<2xui64>, tensor<1xf64>) {
    %a:2 = impulse.sample @draw(%rng, %m) {logpdf = @lp_a, symbol = #impulse.symbol<1>, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %ab = stablehlo.broadcast_in_dim %a#1, dims = [] : (tensor<f64>) -> tensor<2xf64>
    %mu = stablehlo.multiply %ab, %xs : tensor<2xf64>
    %m1 = stablehlo.constant dense<0.0> : tensor<1xf64>
    %half = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %z0:2 = impulse.sample(%a#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<10>, name = "z0"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p0 = stablehlo.compare GT, %z0#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %if0:2 = impulse.if (%p0) ({
      %yo0:2 = impulse.sample @draw1(%z0#0, %m1) {logpdf = @lp_out, symbol = #impulse.symbol<20>, name = "yo0"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yo0#0, %yo0#1 : tensor<2xui64>, tensor<1xf64>
    }, {
      %s0 = stablehlo.slice %mu [0:1] : (tensor<2xf64>) -> tensor<1xf64>
      %yi0:2 = impulse.sample @draw1(%z0#0, %s0) {logpdf = @lp_in, symbol = #impulse.symbol<30>, name = "yi0"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yi0#0, %yi0#1 : tensor<2xui64>, tensor<1xf64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<1xf64>)
    %z1:2 = impulse.sample(%if0#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<11>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p1 = stablehlo.compare GT, %z1#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %if1:2 = impulse.if (%p1) ({
      %yo1:2 = impulse.sample @draw1(%z1#0, %m1) {logpdf = @lp_out, symbol = #impulse.symbol<21>, name = "yo1"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yo1#0, %yo1#1 : tensor<2xui64>, tensor<1xf64>
    }, {
      %s1 = stablehlo.slice %mu [1:2] : (tensor<2xf64>) -> tensor<1xf64>
      %yi1:2 = impulse.sample @draw1(%z1#0, %s1) {logpdf = @lp_in3, symbol = #impulse.symbol<31>, name = "yi1"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yi1#0, %yi1#1 : tensor<2xui64>, tensor<1xf64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<1xf64>)
    return %if1#0, %if0#1 : tensor<2xui64>, tensor<1xf64>
  }
  func.func @main(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %xs : tensor<2xf64>, %trace : tensor<1x7xf64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step = stablehlo.constant dense<0.1> : tensor<f64>
    %res:4 = impulse.infer @model(%rng, %m, %logit, %xs) given %trace init_step_size = %step {
      selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<10>], [#impulse.symbol<20>], [#impulse.symbol<30>], [#impulse.symbol<11>], [#impulse.symbol<21>], [#impulse.symbol<31>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, num_samples = 1 : i64, thinning = 1 : i64, num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<2xf64>, tensor<1x7xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// Not isomorphic (abs vs square): each branch site keeps its own impulse.if.
// CHECK-LABEL: func.func @main
// CHECK: ^bb0(%{{arg[0-9]+}}: tensor<f64>):
// CHECK-NOT: tensor<2xi1>
// CHECK: impulse.if
// CHECK-NOT: tensor<2xi1>
// CHECK: impulse.if
// CHECK-NOT: tensor<2xi1>
// CHECK: stablehlo.abs %{{.+}} : tensor<1xf64>
// CHECK-NOT: tensor<2xi1>
// CHECK: hmc_config

// -----

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
    %h = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
    %r = stablehlo.multiply %h, %n : tensor<f64>
    return %r : tensor<f64>
  }
  func.func private @lp_out(%y : tensor<1xf64>, %m : tensor<1xf64>) -> tensor<f64> {
    %c = stablehlo.constant dense<-1.3862943611198906> : tensor<f64>
    return %c : tensor<f64>
  }
  func.func private @lp_inlog(%y : tensor<1xf64>, %m : tensor<1xf64>) -> tensor<f64> {
    %d = stablehlo.subtract %y, %m : tensor<1xf64>
    %l = stablehlo.log %d : tensor<1xf64>
    %r = stablehlo.reshape %l : (tensor<1xf64>) -> tensor<f64>
    return %r : tensor<f64>
  }
  func.func @model(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %xs : tensor<2xf64>) -> (tensor<2xui64>, tensor<1xf64>) {
    %a:2 = impulse.sample @draw(%rng, %m) {logpdf = @lp_a, symbol = #impulse.symbol<1>, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %ab = stablehlo.broadcast_in_dim %a#1, dims = [] : (tensor<f64>) -> tensor<2xf64>
    %mu = stablehlo.multiply %ab, %xs : tensor<2xf64>
    %m1 = stablehlo.constant dense<0.0> : tensor<1xf64>
    %half = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %z0:2 = impulse.sample(%a#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<10>, name = "z0"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p0 = stablehlo.compare GT, %z0#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %if0:2 = impulse.if (%p0) ({
      %yo0:2 = impulse.sample @draw1(%z0#0, %m1) {logpdf = @lp_out, symbol = #impulse.symbol<20>, name = "yo0"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yo0#0, %yo0#1 : tensor<2xui64>, tensor<1xf64>
    }, {
      %s0 = stablehlo.slice %mu [0:1] : (tensor<2xf64>) -> tensor<1xf64>
      %yi0:2 = impulse.sample @draw1(%z0#0, %s0) {logpdf = @lp_inlog, symbol = #impulse.symbol<30>, name = "yi0"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yi0#0, %yi0#1 : tensor<2xui64>, tensor<1xf64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<1xf64>)
    %z1:2 = impulse.sample(%if0#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<11>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p1 = stablehlo.compare GT, %z1#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %if1:2 = impulse.if (%p1) ({
      %yo1:2 = impulse.sample @draw1(%z1#0, %m1) {logpdf = @lp_out, symbol = #impulse.symbol<21>, name = "yo1"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yo1#0, %yo1#1 : tensor<2xui64>, tensor<1xf64>
    }, {
      %s1 = stablehlo.slice %mu [1:2] : (tensor<2xf64>) -> tensor<1xf64>
      %yi1:2 = impulse.sample @draw1(%z1#0, %s1) {logpdf = @lp_inlog, symbol = #impulse.symbol<31>, name = "yi1"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yi1#0, %yi1#1 : tensor<2xui64>, tensor<1xf64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<1xf64>)
    return %if1#0, %if0#1 : tensor<2xui64>, tensor<1xf64>
  }
  func.func @main(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %xs : tensor<2xf64>, %trace : tensor<1x7xf64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step = stablehlo.constant dense<0.1> : tensor<f64>
    %res:4 = impulse.infer @model(%rng, %m, %logit, %xs) given %trace init_step_size = %step {
      selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<10>], [#impulse.symbol<20>], [#impulse.symbol<30>], [#impulse.symbol<11>], [#impulse.symbol<21>], [#impulse.symbol<31>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, num_samples = 1 : i64, thinning = 1 : i64, num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<2xf64>, tensor<1x7xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// Not vectorized: run on every element, the arm's log could give -inf or NaN.
// CHECK-LABEL: func.func @main
// CHECK: ^bb0(%{{arg[0-9]+}}: tensor<f64>):
// CHECK-NOT: tensor<2xi1>
// CHECK: impulse.if
// CHECK: stablehlo.log %{{.+}} : tensor<1xf64>
// CHECK-NOT: tensor<2xi1>
// CHECK: impulse.if
// CHECK: stablehlo.log %{{.+}} : tensor<1xf64>
// CHECK-NOT: tensor<2xi1>
// CHECK: hmc_config

// -----

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
    %h = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
    %r = stablehlo.multiply %h, %n : tensor<f64>
    return %r : tensor<f64>
  }
  func.func private @lp_out(%y : tensor<1xf64>, %m : tensor<1xf64>) -> tensor<f64> {
    %c = stablehlo.constant dense<-1.3862943611198906> : tensor<f64>
    return %c : tensor<f64>
  }
  func.func private @lp_in(%y : tensor<1xf64>, %m : tensor<1xf64>) -> tensor<f64> {
    %d = stablehlo.subtract %y, %m : tensor<1xf64>
    %n = stablehlo.multiply %d, %d : tensor<1xf64>
    %h = stablehlo.constant dense<-5.000000e-01> : tensor<1xf64>
    %q = stablehlo.multiply %h, %n : tensor<1xf64>
    %r = stablehlo.reshape %q : (tensor<1xf64>) -> tensor<f64>
    return %r : tensor<f64>
  }
  func.func @model(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %xs : tensor<2xf64>) -> (tensor<2xui64>, tensor<1xf64>) {
    %a:2 = impulse.sample @draw(%rng, %m) {logpdf = @lp_a, symbol = #impulse.symbol<1>, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %ab = stablehlo.broadcast_in_dim %a#1, dims = [] : (tensor<f64>) -> tensor<2xf64>
    %mu = stablehlo.multiply %ab, %xs : tensor<2xf64>
    %m1 = stablehlo.constant dense<0.0> : tensor<1xf64>
    %half = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %z0:2 = impulse.sample(%a#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<10>, name = "z0"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p0 = stablehlo.compare GT, %z0#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %if0:2 = impulse.if (%p0) ({
      %yo0:2 = impulse.sample @draw1(%z0#0, %m1) {logpdf = @lp_out, symbol = #impulse.symbol<20>, name = "yo0"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yo0#0, %yo0#1 : tensor<2xui64>, tensor<1xf64>
    }, {
      %s0 = stablehlo.slice %mu [0:1] : (tensor<2xf64>) -> tensor<1xf64>
      %yi0:2 = impulse.sample @draw1(%z0#0, %s0) {logpdf = @lp_in, symbol = #impulse.symbol<30>, name = "yi0"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yi0#0, %yi0#1 : tensor<2xui64>, tensor<1xf64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<1xf64>)
    %z1:2 = impulse.sample(%if0#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<11>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p1 = stablehlo.compare GT, %z1#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %if1:2 = impulse.if (%p1) ({
      %yo1:2 = impulse.sample @draw1(%z1#0, %m1) {logpdf = @lp_out, symbol = #impulse.symbol<21>, name = "yo1"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yo1#0, %yo1#1 : tensor<2xui64>, tensor<1xf64>
    }, {
      %s1 = stablehlo.slice %mu [1:2] : (tensor<2xf64>) -> tensor<1xf64>
      %yi1:2 = impulse.sample @draw1(%z1#0, %s1) {logpdf = @lp_in, symbol = #impulse.symbol<31>, name = "yi1"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %yi1#0, %yi1#1 : tensor<2xui64>, tensor<1xf64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<1xf64>)
    return %if1#0, %if0#1 : tensor<2xui64>, tensor<1xf64>
  }
  func.func @main(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %xs : tensor<2xf64>, %trace : tensor<1x7xf64>, %obs : tensor<1x4xf64>, %w : tensor<f64>) -> tensor<1x7xf64> {
    %res:5 = impulse.infer @model(%rng, %m, %logit, %xs) given %trace constrained_by %obs state = [%w]
        { gibbs_config = #impulse.gibbs_config, selection = [[#impulse.symbol<10>]], all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<10>], [#impulse.symbol<20>], [#impulse.symbol<30>], [#impulse.symbol<11>], [#impulse.symbol<21>], [#impulse.symbol<31>]],
          constrained_addresses = [[#impulse.symbol<20>], [#impulse.symbol<30>], [#impulse.symbol<21>], [#impulse.symbol<31>]], state_schema = ["weight"], num_samples = 1 : i64 }
        : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<1x7xf64>, tensor<1x4xf64>) -> (tensor<1x7xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0 : tensor<1x7xf64>
  }
}

// A Gibbs region reads per-site branch terms, so its branch sites are kept.
// CHECK-LABEL: func.func @main
// CHECK: ^bb0(%{{arg[0-9]+}}: tensor<f64>):
// CHECK-NOT: tensor<2xi1>
// CHECK: impulse.if
// CHECK-NOT: tensor<2xi1>
// CHECK: impulse.if
// CHECK-NOT: tensor<2xi1>
// CHECK: gibbs_config

// -----

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
    %h = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
    %r = stablehlo.multiply %h, %n : tensor<f64>
    return %r : tensor<f64>
  }
  func.func private @lp_out(%y : tensor<1xf64>, %m : tensor<1xf64>) -> tensor<f64> {
    %c = stablehlo.constant dense<-1.3862943611198906> : tensor<f64>
    return %c : tensor<f64>
  }
  func.func private @lp_in(%y : tensor<1xf64>, %m : tensor<1xf64>) -> tensor<f64> {
    %d = stablehlo.subtract %y, %m : tensor<1xf64>
    %n = stablehlo.multiply %d, %d : tensor<1xf64>
    %h = stablehlo.constant dense<-5.000000e-01> : tensor<1xf64>
    %q = stablehlo.multiply %h, %n : tensor<1xf64>
    %r = stablehlo.reshape %q : (tensor<1xf64>) -> tensor<f64>
    return %r : tensor<f64>
  }
  func.func private @lp_in2(%y : tensor<1xf64>, %m : tensor<1xf64>) -> tensor<f64> {
    %d = stablehlo.subtract %y, %m : tensor<1xf64>
    %n = stablehlo.multiply %d, %d : tensor<1xf64>
    %r = stablehlo.reshape %n : (tensor<1xf64>) -> tensor<f64>
    return %r : tensor<f64>
  }
  func.func @model(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %xs : tensor<2xf64>) -> (tensor<2xui64>, tensor<1xf64>) {
    %a:2 = impulse.sample @draw(%rng, %m) {logpdf = @lp_a, symbol = #impulse.symbol<1>, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %ab = stablehlo.broadcast_in_dim %a#1, dims = [] : (tensor<f64>) -> tensor<2xf64>
    %mu = stablehlo.multiply %ab, %xs : tensor<2xf64>
    %m1 = stablehlo.constant dense<0.0> : tensor<1xf64>
    %half = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %z0:2 = impulse.sample(%a#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<10>, name = "z0"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p0 = stablehlo.compare GT, %z0#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %A0:2 = impulse.if (%p0) ({
      %Ao0:2 = impulse.sample @draw1(%z0#0, %m1) {logpdf = @lp_out, symbol = #impulse.symbol<30>, name = "Ao0"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %Ao0#0, %Ao0#1 : tensor<2xui64>, tensor<1xf64>
    }, {
      %s0 = stablehlo.slice %mu [0:1] : (tensor<2xf64>) -> tensor<1xf64>
      %Ai0:2 = impulse.sample @draw1(%z0#0, %s0) {logpdf = @lp_in, symbol = #impulse.symbol<40>, name = "Ai0"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %Ai0#0, %Ai0#1 : tensor<2xui64>, tensor<1xf64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<1xf64>)
    %z1:2 = impulse.sample(%A0#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<11>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p1 = stablehlo.compare GT, %z1#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %A1:2 = impulse.if (%p1) ({
      %Ao1:2 = impulse.sample @draw1(%z1#0, %m1) {logpdf = @lp_out, symbol = #impulse.symbol<31>, name = "Ao1"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %Ao1#0, %Ao1#1 : tensor<2xui64>, tensor<1xf64>
    }, {
      %s1 = stablehlo.slice %mu [1:2] : (tensor<2xf64>) -> tensor<1xf64>
      %Ai1:2 = impulse.sample @draw1(%z1#0, %s1) {logpdf = @lp_in, symbol = #impulse.symbol<41>, name = "Ai1"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %Ai1#0, %Ai1#1 : tensor<2xui64>, tensor<1xf64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<1xf64>)
    %B0:2 = impulse.if (%p0) ({
      %Bo0:2 = impulse.sample @draw1(%A1#0, %m1) {logpdf = @lp_out, symbol = #impulse.symbol<70>, name = "Bo0"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %Bo0#0, %Bo0#1 : tensor<2xui64>, tensor<1xf64>
    }, {
      %Bi0:2 = impulse.sample @draw1(%A1#0, %A0#1) {logpdf = @lp_in2, symbol = #impulse.symbol<80>, name = "Bi0"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %Bi0#0, %Bi0#1 : tensor<2xui64>, tensor<1xf64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<1xf64>)
    %B1:2 = impulse.if (%p1) ({
      %Bo1:2 = impulse.sample @draw1(%B0#0, %m1) {logpdf = @lp_out, symbol = #impulse.symbol<71>, name = "Bo1"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %Bo1#0, %Bo1#1 : tensor<2xui64>, tensor<1xf64>
    }, {
      %Bi1:2 = impulse.sample @draw1(%B0#0, %A1#1) {logpdf = @lp_in2, symbol = #impulse.symbol<81>, name = "Bi1"} : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)
      impulse.yield %Bi1#0, %Bi1#1 : tensor<2xui64>, tensor<1xf64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<1xf64>)
    return %B1#0, %B0#1 : tensor<2xui64>, tensor<1xf64>
  }
  func.func @main(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %xs : tensor<2xf64>, %trace : tensor<1x11xf64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step = stablehlo.constant dense<0.1> : tensor<f64>
    %res:4 = impulse.infer @model(%rng, %m, %logit, %xs) given %trace init_step_size = %step {
      selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<10>], [#impulse.symbol<30>], [#impulse.symbol<40>], [#impulse.symbol<11>], [#impulse.symbol<31>], [#impulse.symbol<41>], [#impulse.symbol<70>], [#impulse.symbol<80>], [#impulse.symbol<71>], [#impulse.symbol<81>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, num_samples = 1 : i64, thinning = 1 : i64, num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<2xf64>, tensor<1x11xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// Two families; the second reads the first's selected vector.
// CHECK-LABEL: func.func @main
// CHECK: %[[P:.+]] = stablehlo.concatenate %{{.+}}, %{{.+}}, dim = 0 : (tensor<1xi1>, tensor<1xi1>) -> tensor<2xi1>
// CHECK-NEXT: %[[YO:.+]] = stablehlo.concatenate %{{.+}}, %{{.+}}, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[YI:.+]] = stablehlo.concatenate %{{.+}}, %{{.+}}, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
// CHECK: %[[Y:.+]] = stablehlo.select %[[P]], %[[YO]], %[[YI]] : tensor<2xi1>, tensor<2xf64>
// CHECK: %[[PB:.+]] = stablehlo.concatenate %{{.+}}, %{{.+}}, dim = 0 : (tensor<1xi1>, tensor<1xi1>) -> tensor<2xi1>
// CHECK-NEXT: %[[WI:.+]] = stablehlo.concatenate %{{.+}}, %{{.+}}, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[DB:.+]] = stablehlo.subtract %[[WI]], %[[Y]] : tensor<2xf64>
// CHECK-NEXT: %{{.+}} = stablehlo.multiply %[[DB]], %[[DB]] : tensor<2xf64>
// CHECK: ^bb0(%{{arg[0-9]+}}: tensor<f64>):
// CHECK-NOT: impulse.if
// CHECK: hmc_config
