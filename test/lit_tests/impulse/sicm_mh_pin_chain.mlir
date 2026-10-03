// RUN: enzymexlamlir-opt %s --inline-infer-regions --canonicalize --cse --sicm --cse | FileCheck %s

// A pin walks back through preceding MH regions that do not select the site.

module {
  func.func private @draw(%rng : tensor<2xui64>, %m : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %m : tensor<2xui64>, tensor<f64>
  }
  func.func private @lp(%x : tensor<f64>, %m : tensor<f64>) -> tensor<f64> {
    %d = stablehlo.subtract %x, %m : tensor<f64>
    %n = stablehlo.multiply %d, %d : tensor<f64>
    %r = stablehlo.negate %n : tensor<f64>
    return %r : tensor<f64>
  }
  func.func private @obs_lp(%y : tensor<f64>, %m : tensor<f64>) -> tensor<f64> {
    %l = chlo.lgamma %y : tensor<f64> -> tensor<f64>
    %d = stablehlo.subtract %m, %l : tensor<f64>
    return %d : tensor<f64>
  }

  func.func @model(%rng : tensor<2xui64>, %m : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %a:2 = impulse.sample @draw(%rng, %m) {logpdf = @lp, symbol = #impulse.symbol<1>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %b:2 = impulse.sample @draw(%a#0, %a#1) {logpdf = @lp, symbol = #impulse.symbol<2>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %c:2 = impulse.sample @draw(%b#0, %b#1) {logpdf = @lp, symbol = #impulse.symbol<3>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %y:2 = impulse.sample @draw(%c#0, %c#1) {logpdf = @obs_lp, symbol = #impulse.symbol<4>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<f64>
  }

  // CHECK-LABEL: func.func @sweep(
  // CHECK-SAME: %[[T0:[^:]+]]: tensor<1x4xf64>, %{{[^:]+}}: tensor<f64>)
  // CHECK: %{{.+}} = impulse.slice %[[T0]] {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
  // CHECK: %{{.+}} = impulse.slice %[[T0]] {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 2>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
  // CHECK: %[[Y0:.+]] = impulse.slice %[[T0]] {limit_indices = array<i64: 1, 4>, start_indices = array<i64: 0, 3>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
  // CHECK-NEXT: %[[Y:.+]] = impulse.reshape %[[Y0]] : (tensor<1x1xf64>) -> tensor<f64>
  // CHECK: %[[LG:.+]] = chlo.lgamma %[[Y]] : tensor<f64> -> tensor<f64>
  // CHECK-NEXT: %[[YLP:.+]] = stablehlo.subtract %{{.+}}, %[[LG]] : tensor<f64>
  // CHECK-NEXT: %[[T1:[^,]+]], {{.+}} = impulse.infer_region @model(%{{.+}}) given %[[T0]] state
  // CHECK: impulse.yield %{{.+}}, %[[YLP]] : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
  // CHECK: %{{.+}} = impulse.slice %[[T1]] {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
  // CHECK-NOT: impulse.slice
  // CHECK: %[[T2:[^,]+]], {{.+}} = impulse.infer_region @model(%{{.+}}) given %[[T1]] state
  // CHECK: impulse.yield %{{.+}}, %[[YLP]] : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
  // CHECK: %{{.+}} = impulse.slice %[[T2]] {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
  // CHECK-NOT: impulse.slice
  // CHECK-NOT: chlo.lgamma
  // CHECK: impulse.infer_region @model(%{{.+}}) given %[[T2]] state
  // CHECK: %[[YLP3:.+]] = stablehlo.subtract %{{.+}}, %[[LG]] : tensor<f64>
  // CHECK: impulse.yield %{{.+}}, %[[YLP3]] : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
  func.func @sweep(%rng : tensor<2xui64>, %m : tensor<f64>, %trace : tensor<1x4xf64>, %w : tensor<f64>) -> (tensor<1x4xf64>, tensor<f64>) {
    %r1:5 = impulse.infer @model(%rng, %m) given %trace state = [%w] {
      mh_config = #impulse.mh_config, selection = [[#impulse.symbol<1>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
      state_schema = ["weight"], num_samples = 1 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    %r2:5 = impulse.infer @model(%r1#3, %m) given %r1#0 state = [%r1#4] {
      mh_config = #impulse.mh_config, selection = [[#impulse.symbol<2>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
      state_schema = ["weight"], num_samples = 1 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    %r3:5 = impulse.infer @model(%r2#3, %m) given %r2#0 state = [%r2#4] {
      mh_config = #impulse.mh_config, selection = [[#impulse.symbol<3>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
      state_schema = ["weight"], num_samples = 1 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %r3#0, %r3#4 : tensor<1x4xf64>, tensor<f64>
  }

  // An op on the trace between moves 1 and 2 stops the walk.
  // CHECK-LABEL: func.func @sweep_stops(
  // CHECK-SAME: %[[T0:[^:]+]]: tensor<1x4xf64>, %{{[^:]+}}: tensor<f64>, %[[S:[^:]+]]: tensor<1x4xf64>)
  // CHECK: chlo.lgamma
  // CHECK: %[[T1:[^,]+]], {{.+}} = impulse.infer_region @model(%{{.+}}) given %[[T0]] state
  // CHECK: %[[TS:.+]] = stablehlo.multiply %[[T1]], %[[S]] : tensor<1x4xf64>
  // CHECK-NEXT: %{{.+}} = impulse.slice %[[TS]] {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
  // CHECK-NEXT: impulse.reshape
  // CHECK-NEXT: %{{.+}} = impulse.slice %[[TS]] {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 2>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
  // CHECK-NEXT: impulse.reshape
  // CHECK-NEXT: %{{.+}} = impulse.slice %[[TS]] {limit_indices = array<i64: 1, 4>, start_indices = array<i64: 0, 3>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
  // CHECK: chlo.lgamma
  // CHECK: %[[T2:[^,]+]], {{.+}} = impulse.infer_region @model(%{{.+}}) given %[[TS]] state
  // CHECK: %{{.+}} = impulse.slice %[[T2]] {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
  // CHECK-NOT: impulse.slice
  // CHECK-NOT: chlo.lgamma
  // CHECK: impulse.infer_region @model(%{{.+}}) given %[[T2]] state
  func.func @sweep_stops(%rng : tensor<2xui64>, %m : tensor<f64>, %trace : tensor<1x4xf64>, %w : tensor<f64>, %s : tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<f64>) {
    %r1:5 = impulse.infer @model(%rng, %m) given %trace state = [%w] {
      mh_config = #impulse.mh_config, selection = [[#impulse.symbol<1>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
      state_schema = ["weight"], num_samples = 1 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    %t = stablehlo.multiply %r1#0, %s : tensor<1x4xf64>
    %r2:5 = impulse.infer @model(%r1#3, %m) given %t state = [%r1#4] {
      mh_config = #impulse.mh_config, selection = [[#impulse.symbol<2>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
      state_schema = ["weight"], num_samples = 1 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    %r3:5 = impulse.infer @model(%r2#3, %m) given %r2#0 state = [%r2#4] {
      mh_config = #impulse.mh_config, selection = [[#impulse.symbol<3>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
      state_schema = ["weight"], num_samples = 1 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %r3#0, %r3#4 : tensor<1x4xf64>, tensor<f64>
  }
}
