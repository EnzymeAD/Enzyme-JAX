// RUN: enzymexlamlir-opt %s --split-input-file --inline-infer-regions --sicm | FileCheck %s

module {
  func.func private @draw(%rng : tensor<2xui64>, %m : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %m : tensor<2xui64>, tensor<f64>
  }
  func.func private @lp_n(%x : tensor<f64>, %m : tensor<f64>) -> tensor<f64> {
    %d = stablehlo.subtract %x, %m : tensor<f64>
    %n = stablehlo.multiply %d, %d : tensor<f64>
    %h = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
    %r = stablehlo.multiply %h, %n : tensor<f64>
    return %r : tensor<f64>
  }
  func.func private @lp_y(%y : tensor<f64>, %mu : tensor<f64>, %sd : tensor<f64>) -> tensor<f64> {
    %d = stablehlo.subtract %y, %mu : tensor<f64>
    %z = stablehlo.divide %d, %sd : tensor<f64>
    %q = stablehlo.multiply %z, %z : tensor<f64>
    %h = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
    %r = stablehlo.multiply %h, %q : tensor<f64>
    %l = stablehlo.log %sd : tensor<f64>
    %t = stablehlo.subtract %r, %l : tensor<f64>
    return %t : tensor<f64>
  }
  func.func private @lp_y2(%y : tensor<f64>, %mu : tensor<f64>, %sd : tensor<f64>) -> tensor<f64> {
    %d = stablehlo.subtract %y, %mu : tensor<f64>
    %z = stablehlo.divide %d, %sd : tensor<f64>
    %q = stablehlo.abs %z : tensor<f64>
    %l = stablehlo.log %sd : tensor<f64>
    %t = stablehlo.subtract %q, %l : tensor<f64>
    return %t : tensor<f64>
  }
  func.func @model(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %s : tensor<f64>, %xs : tensor<4xf64>) -> tensor<2xui64> {
    %a:2 = impulse.sample @draw(%rng, %m) {logpdf = @lp_n, symbol = #impulse.symbol<1>, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %b:2 = impulse.sample @draw(%a#0, %m) {logpdf = @lp_n, symbol = #impulse.symbol<2>, name = "b"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %half = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %ten = stablehlo.constant dense<1.000000e+01> : tensor<f64>
    %z0:2 = impulse.sample(%b#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<10>, name = "z0"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p0 = stablehlo.compare GT, %z0#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %xs0 = stablehlo.slice %xs [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %x0 = stablehlo.reshape %xs0 : (tensor<1xf64>) -> tensor<f64>
    %ax0 = stablehlo.multiply %x0, %a#1 : tensor<f64>
    %lin0 = stablehlo.add %ax0, %b#1 : tensor<f64>
    %mu0 = stablehlo.select %p0, %zero, %lin0 : tensor<i1>, tensor<f64>
    %sd0 = stablehlo.select %p0, %ten, %s : tensor<i1>, tensor<f64>
    %y0:2 = impulse.sample @draw(%z0#0, %mu0, %sd0) {logpdf = @lp_y, symbol = #impulse.symbol<20>, name = "y0"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %z1:2 = impulse.sample(%y0#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<11>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p1 = stablehlo.compare GT, %z1#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %xs1 = stablehlo.slice %xs [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %x1 = stablehlo.reshape %xs1 : (tensor<1xf64>) -> tensor<f64>
    %ax1 = stablehlo.multiply %x1, %a#1 : tensor<f64>
    %lin1 = stablehlo.add %ax1, %b#1 : tensor<f64>
    %mu1 = stablehlo.select %p1, %zero, %lin1 : tensor<i1>, tensor<f64>
    %sd1 = stablehlo.select %p1, %ten, %s : tensor<i1>, tensor<f64>
    %y1:2 = impulse.sample @draw(%z1#0, %mu1, %sd1) {logpdf = @lp_y, symbol = #impulse.symbol<21>, name = "y1"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %z2:2 = impulse.sample(%y1#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<12>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p2 = stablehlo.compare GT, %z2#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %xs2 = stablehlo.slice %xs [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %x2 = stablehlo.reshape %xs2 : (tensor<1xf64>) -> tensor<f64>
    %ax2 = stablehlo.multiply %x2, %a#1 : tensor<f64>
    %lin2 = stablehlo.add %ax2, %b#1 : tensor<f64>
    %mu2 = stablehlo.select %p2, %zero, %lin2 : tensor<i1>, tensor<f64>
    %sd2 = stablehlo.select %p2, %ten, %s : tensor<i1>, tensor<f64>
    %y2:2 = impulse.sample @draw(%z2#0, %mu2, %sd2) {logpdf = @lp_y, symbol = #impulse.symbol<22>, name = "y2"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %z3:2 = impulse.sample(%y2#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<13>, name = "z3"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p3 = stablehlo.compare GT, %z3#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %xs3 = stablehlo.slice %xs [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %x3 = stablehlo.reshape %xs3 : (tensor<1xf64>) -> tensor<f64>
    %ax3 = stablehlo.multiply %x3, %a#1 : tensor<f64>
    %lin3 = stablehlo.add %ax3, %b#1 : tensor<f64>
    %mu3 = stablehlo.select %p3, %zero, %lin3 : tensor<i1>, tensor<f64>
    %sd3 = stablehlo.select %p3, %ten, %s : tensor<i1>, tensor<f64>
    %y3:2 = impulse.sample @draw(%z3#0, %mu3, %sd3) {logpdf = @lp_y, symbol = #impulse.symbol<23>, name = "y3"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %y3#0 : tensor<2xui64>
  }
  func.func @main(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %s : tensor<f64>, %xs : tensor<4xf64>, %trace : tensor<1x20xf64>, %w : tensor<f64>) -> tensor<1x20xf64> {
    %res:5 = impulse.infer @model(%rng, %m, %logit, %s, %xs) given %trace state = [%w] {
      mh_config = #impulse.mh_config, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]], all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<10>], [#impulse.symbol<20>], [#impulse.symbol<11>], [#impulse.symbol<21>], [#impulse.symbol<12>], [#impulse.symbol<22>], [#impulse.symbol<13>], [#impulse.symbol<23>]],
      state_schema = ["weight"], num_samples = 1 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<4xf64>, tensor<f64>, tensor<1x20xf64>) -> (tensor<1x20xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0 : tensor<1x20xf64>
  }
}

// Block move on a and b: the four dirty y_k terms form one family.
// CHECK-LABEL: func.func @main
// CHECK: %[[X:.+]] = stablehlo.slice %{{.+}} [0:4] : (tensor<4xf64>) -> tensor<4xf64>
// CHECK: %[[P:.+]] = stablehlo.concatenate %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, dim = 0 : (tensor<1xi1>, tensor<1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<4xi1>
// CHECK-NEXT: %[[YS:.+]] = stablehlo.slice %{{.+}} [0:1, 3:10:2] : (tensor<1x20xf64>) -> tensor<1x4xf64>
// CHECK-NEXT: %[[Y:.+]] = stablehlo.reshape %[[YS]] : (tensor<1x4xf64>) -> tensor<4xf64>
// CHECK: %[[SD:.+]] = stablehlo.concatenate %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
// CHECK: %[[LSD:.+]] = stablehlo.concatenate %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
// CHECK: ^bb0(%[[A:arg[0-9]+]]: tensor<f64>, %[[B:arg[0-9]+]]: tensor<f64>):
// CHECK: %[[AB:.+]] = stablehlo.broadcast_in_dim %[[A]], dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT: %[[AX:.+]] = stablehlo.multiply %[[X]], %[[AB]] : tensor<4xf64>
// CHECK-NEXT: %[[BB:.+]] = stablehlo.broadcast_in_dim %[[B]], dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT: %[[LIN:.+]] = stablehlo.add %[[AX]], %[[BB]] : tensor<4xf64>
// CHECK-NEXT: %[[ZB:.+]] = stablehlo.broadcast_in_dim %{{.+}}, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT: %[[MU:.+]] = stablehlo.select %[[P]], %[[ZB]], %[[LIN]] : tensor<4xi1>, tensor<4xf64>
// CHECK-NEXT: %[[D:.+]] = stablehlo.subtract %[[Y]], %[[MU]] : tensor<4xf64>
// CHECK-NEXT: %[[Z:.+]] = stablehlo.divide %[[D]], %[[SD]] : tensor<4xf64>
// CHECK-NEXT: %[[Q:.+]] = stablehlo.multiply %[[Z]], %[[Z]] : tensor<4xf64>
// CHECK-NEXT: %[[HB:.+]] = stablehlo.broadcast_in_dim %{{.+}}, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT: %[[R:.+]] = stablehlo.multiply %[[HB]], %[[Q]] : tensor<4xf64>
// CHECK-NEXT: %[[T:.+]] = stablehlo.subtract %[[R]], %[[LSD]] : tensor<4xf64>
// CHECK-NEXT: %[[T0:.+]] = impulse.slice %[[T]] {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<4xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[Y0:.+]] = impulse.reshape %[[T0]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[T1:.+]] = impulse.slice %[[T]] {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<4xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[Y1:.+]] = impulse.reshape %[[T1]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[T2:.+]] = impulse.slice %[[T]] {limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>} : (tensor<4xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[Y2:.+]] = impulse.reshape %[[T2]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[T3:.+]] = impulse.slice %[[T]] {limit_indices = array<i64: 4>, start_indices = array<i64: 3>, strides = array<i64: 1>} : (tensor<4xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[Y3:.+]] = impulse.reshape %[[T3]] : (tensor<1xf64>) -> tensor<f64>
// CHECK: impulse.yield %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[Y0]], %{{.+}}, %[[Y1]], %{{.+}}, %[[Y2]], %{{.+}}, %[[Y3]] :
// CHECK: mh_config

// -----

module {
  func.func private @draw(%rng : tensor<2xui64>, %m : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %m : tensor<2xui64>, tensor<f64>
  }
  func.func private @lp_n(%x : tensor<f64>, %m : tensor<f64>) -> tensor<f64> {
    %d = stablehlo.subtract %x, %m : tensor<f64>
    %n = stablehlo.multiply %d, %d : tensor<f64>
    %h = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
    %r = stablehlo.multiply %h, %n : tensor<f64>
    return %r : tensor<f64>
  }
  func.func private @lp_y(%y : tensor<f64>, %mu : tensor<f64>, %sd : tensor<f64>) -> tensor<f64> {
    %d = stablehlo.subtract %y, %mu : tensor<f64>
    %z = stablehlo.divide %d, %sd : tensor<f64>
    %q = stablehlo.multiply %z, %z : tensor<f64>
    %h = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
    %r = stablehlo.multiply %h, %q : tensor<f64>
    %l = stablehlo.log %sd : tensor<f64>
    %t = stablehlo.subtract %r, %l : tensor<f64>
    return %t : tensor<f64>
  }
  func.func private @lp_y2(%y : tensor<f64>, %mu : tensor<f64>, %sd : tensor<f64>) -> tensor<f64> {
    %d = stablehlo.subtract %y, %mu : tensor<f64>
    %z = stablehlo.divide %d, %sd : tensor<f64>
    %q = stablehlo.abs %z : tensor<f64>
    %l = stablehlo.log %sd : tensor<f64>
    %t = stablehlo.subtract %q, %l : tensor<f64>
    return %t : tensor<f64>
  }
  func.func @model(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %s : tensor<f64>, %xs : tensor<4xf64>) -> tensor<2xui64> {
    %a:2 = impulse.sample @draw(%rng, %m) {logpdf = @lp_n, symbol = #impulse.symbol<1>, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %b:2 = impulse.sample @draw(%a#0, %m) {logpdf = @lp_n, symbol = #impulse.symbol<2>, name = "b"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %half = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %ten = stablehlo.constant dense<1.000000e+01> : tensor<f64>
    %z0:2 = impulse.sample(%b#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<10>, name = "z0"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p0 = stablehlo.compare GT, %z0#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %xs0 = stablehlo.slice %xs [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %x0 = stablehlo.reshape %xs0 : (tensor<1xf64>) -> tensor<f64>
    %ax0 = stablehlo.multiply %x0, %a#1 : tensor<f64>
    %lin0 = stablehlo.add %ax0, %b#1 : tensor<f64>
    %mu0 = stablehlo.select %p0, %zero, %lin0 : tensor<i1>, tensor<f64>
    %sd0 = stablehlo.select %p0, %ten, %s : tensor<i1>, tensor<f64>
    %y0:2 = impulse.sample @draw(%z0#0, %mu0, %sd0) {logpdf = @lp_y, symbol = #impulse.symbol<20>, name = "y0"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %z1:2 = impulse.sample(%y0#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<11>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p1 = stablehlo.compare GT, %z1#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %xs1 = stablehlo.slice %xs [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %x1 = stablehlo.reshape %xs1 : (tensor<1xf64>) -> tensor<f64>
    %ax1 = stablehlo.multiply %x1, %a#1 : tensor<f64>
    %lin1 = stablehlo.add %ax1, %b#1 : tensor<f64>
    %mu1 = stablehlo.select %p1, %zero, %lin1 : tensor<i1>, tensor<f64>
    %sd1 = stablehlo.select %p1, %ten, %s : tensor<i1>, tensor<f64>
    %y1:2 = impulse.sample @draw(%z1#0, %mu1, %sd1) {logpdf = @lp_y, symbol = #impulse.symbol<21>, name = "y1"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %z2:2 = impulse.sample(%y1#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<12>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p2 = stablehlo.compare GT, %z2#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %xs2 = stablehlo.slice %xs [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %x2 = stablehlo.reshape %xs2 : (tensor<1xf64>) -> tensor<f64>
    %ax2 = stablehlo.multiply %x2, %a#1 : tensor<f64>
    %lin2 = stablehlo.add %ax2, %b#1 : tensor<f64>
    %mu2 = stablehlo.select %p2, %zero, %lin2 : tensor<i1>, tensor<f64>
    %sd2 = stablehlo.select %p2, %ten, %s : tensor<i1>, tensor<f64>
    %y2:2 = impulse.sample @draw(%z2#0, %mu2, %sd2) {logpdf = @lp_y, symbol = #impulse.symbol<22>, name = "y2"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %z3:2 = impulse.sample(%y2#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<13>, name = "z3"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p3 = stablehlo.compare GT, %z3#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %xs3 = stablehlo.slice %xs [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %x3 = stablehlo.reshape %xs3 : (tensor<1xf64>) -> tensor<f64>
    %ax3 = stablehlo.multiply %x3, %a#1 : tensor<f64>
    %lin3 = stablehlo.add %ax3, %b#1 : tensor<f64>
    %mu3 = stablehlo.select %p3, %zero, %lin3 : tensor<i1>, tensor<f64>
    %sd3 = stablehlo.select %p3, %ten, %s : tensor<i1>, tensor<f64>
    %y3:2 = impulse.sample @draw(%z3#0, %mu3, %sd3) {logpdf = @lp_y, symbol = #impulse.symbol<23>, name = "y3"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %y3#0 : tensor<2xui64>
  }
  func.func @main(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %s : tensor<f64>, %xs : tensor<4xf64>, %trace : tensor<1x20xf64>, %w : tensor<f64>) -> tensor<1x20xf64> {
    %res:5 = impulse.infer @model(%rng, %m, %logit, %s, %xs) given %trace state = [%w] {
      mh_config = #impulse.mh_config, selection = [[#impulse.symbol<10>]], all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<10>], [#impulse.symbol<20>], [#impulse.symbol<11>], [#impulse.symbol<21>], [#impulse.symbol<12>], [#impulse.symbol<22>], [#impulse.symbol<13>], [#impulse.symbol<23>]],
      state_schema = ["weight"], num_samples = 1 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<4xf64>, tensor<f64>, tensor<1x20xf64>) -> (tensor<1x20xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0 : tensor<1x20xf64>
  }
}

// Single-site move on z_0: only y_0 is dirty, so there is no family.
// CHECK-LABEL: func.func @main
// CHECK: ^bb0(%{{arg[0-9]+}}: tensor<f64>):
// CHECK-NOT: tensor<4x
// CHECK: impulse.yield
// CHECK-NOT: tensor<4x
// CHECK: mh_config

// -----

module {
  func.func private @draw(%rng : tensor<2xui64>, %m : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %m : tensor<2xui64>, tensor<f64>
  }
  func.func private @lp_n(%x : tensor<f64>, %m : tensor<f64>) -> tensor<f64> {
    %d = stablehlo.subtract %x, %m : tensor<f64>
    %n = stablehlo.multiply %d, %d : tensor<f64>
    %h = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
    %r = stablehlo.multiply %h, %n : tensor<f64>
    return %r : tensor<f64>
  }
  func.func private @lp_y(%y : tensor<f64>, %mu : tensor<f64>, %sd : tensor<f64>) -> tensor<f64> {
    %d = stablehlo.subtract %y, %mu : tensor<f64>
    %z = stablehlo.divide %d, %sd : tensor<f64>
    %q = stablehlo.multiply %z, %z : tensor<f64>
    %h = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
    %r = stablehlo.multiply %h, %q : tensor<f64>
    %l = stablehlo.log %sd : tensor<f64>
    %t = stablehlo.subtract %r, %l : tensor<f64>
    return %t : tensor<f64>
  }
  func.func private @lp_y2(%y : tensor<f64>, %mu : tensor<f64>, %sd : tensor<f64>) -> tensor<f64> {
    %d = stablehlo.subtract %y, %mu : tensor<f64>
    %z = stablehlo.divide %d, %sd : tensor<f64>
    %q = stablehlo.abs %z : tensor<f64>
    %l = stablehlo.log %sd : tensor<f64>
    %t = stablehlo.subtract %q, %l : tensor<f64>
    return %t : tensor<f64>
  }
  func.func @model(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %s : tensor<f64>, %xs : tensor<4xf64>) -> tensor<2xui64> {
    %a:2 = impulse.sample @draw(%rng, %m) {logpdf = @lp_n, symbol = #impulse.symbol<1>, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %b:2 = impulse.sample @draw(%a#0, %m) {logpdf = @lp_n, symbol = #impulse.symbol<2>, name = "b"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %half = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %ten = stablehlo.constant dense<1.000000e+01> : tensor<f64>
    %z0:2 = impulse.sample(%b#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<10>, name = "z0"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p0 = stablehlo.compare GT, %z0#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %xs0 = stablehlo.slice %xs [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %x0 = stablehlo.reshape %xs0 : (tensor<1xf64>) -> tensor<f64>
    %ax0 = stablehlo.multiply %x0, %a#1 : tensor<f64>
    %lin0 = stablehlo.add %ax0, %b#1 : tensor<f64>
    %mu0 = stablehlo.select %p0, %zero, %lin0 : tensor<i1>, tensor<f64>
    %sd0 = stablehlo.select %p0, %ten, %s : tensor<i1>, tensor<f64>
    %y0:2 = impulse.sample @draw(%z0#0, %mu0, %sd0) {logpdf = @lp_y, symbol = #impulse.symbol<20>, name = "y0"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %z1:2 = impulse.sample(%y0#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<11>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p1 = stablehlo.compare GT, %z1#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %xs1 = stablehlo.slice %xs [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %x1 = stablehlo.reshape %xs1 : (tensor<1xf64>) -> tensor<f64>
    %ax1 = stablehlo.multiply %x1, %a#1 : tensor<f64>
    %lin1 = stablehlo.add %ax1, %b#1 : tensor<f64>
    %mu1 = stablehlo.select %p1, %zero, %lin1 : tensor<i1>, tensor<f64>
    %sd1 = stablehlo.select %p1, %ten, %s : tensor<i1>, tensor<f64>
    %y1:2 = impulse.sample @draw(%z1#0, %mu1, %sd1) {logpdf = @lp_y, symbol = #impulse.symbol<21>, name = "y1"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %z2:2 = impulse.sample(%y1#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<12>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p2 = stablehlo.compare GT, %z2#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %xs2 = stablehlo.slice %xs [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %x2 = stablehlo.reshape %xs2 : (tensor<1xf64>) -> tensor<f64>
    %ax2 = stablehlo.multiply %x2, %a#1 : tensor<f64>
    %lin2 = stablehlo.add %ax2, %b#1 : tensor<f64>
    %mu2 = stablehlo.select %p2, %zero, %lin2 : tensor<i1>, tensor<f64>
    %sd2 = stablehlo.select %p2, %ten, %s : tensor<i1>, tensor<f64>
    %y2:2 = impulse.sample @draw(%z2#0, %mu2, %sd2) {logpdf = @lp_y2, symbol = #impulse.symbol<22>, name = "y2"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %z3:2 = impulse.sample(%y2#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<13>, name = "z3"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %p3 = stablehlo.compare GT, %z3#1, %half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %xs3 = stablehlo.slice %xs [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %x3 = stablehlo.reshape %xs3 : (tensor<1xf64>) -> tensor<f64>
    %ax3 = stablehlo.multiply %x3, %a#1 : tensor<f64>
    %lin3 = stablehlo.add %ax3, %b#1 : tensor<f64>
    %mu3 = stablehlo.select %p3, %zero, %lin3 : tensor<i1>, tensor<f64>
    %sd3 = stablehlo.select %p3, %ten, %s : tensor<i1>, tensor<f64>
    %y3:2 = impulse.sample @draw(%z3#0, %mu3, %sd3) {logpdf = @lp_y2, symbol = #impulse.symbol<23>, name = "y3"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %y3#0 : tensor<2xui64>
  }
  func.func @main(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %s : tensor<f64>, %xs : tensor<4xf64>, %trace : tensor<1x20xf64>, %w : tensor<f64>) -> tensor<1x20xf64> {
    %res:5 = impulse.infer @model(%rng, %m, %logit, %s, %xs) given %trace state = [%w] {
      mh_config = #impulse.mh_config, selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]], all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<10>], [#impulse.symbol<20>], [#impulse.symbol<11>], [#impulse.symbol<21>], [#impulse.symbol<12>], [#impulse.symbol<22>], [#impulse.symbol<13>], [#impulse.symbol<23>]],
      state_schema = ["weight"], num_samples = 1 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<4xf64>, tensor<f64>, tensor<1x20xf64>) -> (tensor<1x20xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0 : tensor<1x20xf64>
  }
}

// Two families of two, below the minimum family size of four: all scalar.
// CHECK-LABEL: func.func @main
// CHECK: ^bb0(%{{arg[0-9]+}}: tensor<f64>, %{{arg[0-9]+}}: tensor<f64>):
// CHECK-NOT: tensor<4x
// CHECK: impulse.yield
// CHECK-NOT: tensor<4x
// CHECK: mh_config
