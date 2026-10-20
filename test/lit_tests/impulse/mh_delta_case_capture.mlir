// RUN: enzymexlamlir-opt %s --expand-impulse | FileCheck %s

// y is dirty through the case region's captured slope and must be rescored.

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>
  func.func private @y_draw(%rng : tensor<2xui64>, %mu : tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
  func.func private @y_logpdf(%y : tensor<3xf64>, %mu : tensor<3xf64>) -> tensor<f64>

  func.func @model(%rng : tensor<2xui64>, %xs : tensor<3xf64>, %c : tensor<f64>) -> (tensor<2xui64>, tensor<3xf64>) {
    %m:2 = impulse.sample @normal(%rng, %c, %c) <{ logpdf = @logpdf, symbol = #impulse.symbol<1>, name = "m" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %slope:2 = impulse.sample @normal(%m#0, %c, %c) <{ logpdf = @logpdf, symbol = #impulse.symbol<2>, name = "slope" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %amp:2 = impulse.sample @normal(%slope#0, %c, %c) <{ logpdf = @logpdf, symbol = #impulse.symbol<3>, name = "amp" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %idx = stablehlo.convert %m#1 : (tensor<f64>) -> tensor<i32>
    %mu = "stablehlo.case"(%idx) ({
      %ba = stablehlo.broadcast_in_dim %amp#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %sx = stablehlo.sine %xs : tensor<3xf64>
      %ms = stablehlo.multiply %ba, %sx : tensor<3xf64>
      stablehlo.return %ms : tensor<3xf64>
    }, {
      %bs = stablehlo.broadcast_in_dim %slope#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %ml = stablehlo.multiply %bs, %xs : tensor<3xf64>
      stablehlo.return %ml : tensor<3xf64>
    }) : (tensor<i32>) -> tensor<3xf64>
    %y:2 = impulse.sample @y_draw(%amp#0, %mu) <{ logpdf = @y_logpdf, symbol = #impulse.symbol<4>, name = "y" }> : (tensor<2xui64>, tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  func.func @mh(%rng : tensor<2xui64>, %xs : tensor<3xf64>, %c : tensor<f64>, %trace : tensor<1x10xf64>, %weight : tensor<f64>) -> (tensor<1x10xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>) {
    %step:4 = impulse.mh @model(%rng, %xs, %c) given %trace weight %weight
        <{ name = "mh", selection = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]], regenerate_addresses = [[#impulse.symbol<2>]] }>
        : (tensor<1x10xf64>, tensor<f64>, tensor<2xui64>, tensor<3xf64>, tensor<f64>) -> (tensor<1x10xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>)
    return %step#0, %step#1, %step#2, %step#3 : tensor<1x10xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @mh
// CHECK:         %[[REGEN:.+]]:4 = call @model.regenerate(
// CHECK-NEXT:    %[[DSL:.+]] = impulse.slice %[[REGEN]]#1 <start_indices = [0], limit_indices = [1], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %[[DELTA:.+]] = impulse.reshape %[[DSL]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[CSL:.+]] = impulse.slice %[[REGEN]]#1 <start_indices = [1], limit_indices = [2], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %[[CORR:.+]] = impulse.reshape %[[CSL]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[ALPHA:.+]] = arith.subf %[[DELTA]], %[[CORR]] : tensor<f64>

// CHECK-LABEL: func.func @model.regenerate
// CHECK-SAME:      (%[[TR:[^:]+]]: tensor<1x10xf64>, %[[R:[^:]+]]: tensor<2xui64>, %[[XS:[^:]+]]: tensor<3xf64>, %[[C:[^:]+]]: tensor<f64>)
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK:         %[[S:.+]]:2 = call @normal(%[[R]], %[[C]], %[[C]])
// CHECK-NEXT:    %[[SCSSL:.+]] = impulse.slice %[[TR]] <start_indices = [0, 7], limit_indices = [1, 8], strides = [1, 1]> : (tensor<1x10xf64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[SCS:.+]] = impulse.reshape %[[SCSSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[LPS:.+]] = call @logpdf(%[[S]]#1, %[[C]], %[[C]])
// CHECK-NEXT:    %[[DS:.+]] = arith.subf %[[LPS]], %[[SCS]] : tensor<f64>
// CHECK-NEXT:    %[[D1:.+]] = arith.addf %[[DS]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:    %[[CORR:.+]] = arith.addf %[[DS]], %[[ZERO]] : tensor<f64>
// CHECK:         %[[MU:.+]] = "stablehlo.case"(%{{.+}}) ({
// CHECK:         }, {
// CHECK-NEXT:      %[[BS:.+]] = stablehlo.broadcast_in_dim %[[S]]#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
// CHECK-NEXT:      %[[ML:.+]] = stablehlo.multiply %[[BS]], %[[XS]] : tensor<3xf64>
// CHECK-NEXT:      stablehlo.return %[[ML]] : tensor<3xf64>
// CHECK-NEXT:    }) : (tensor<i32>) -> tensor<3xf64>
// CHECK-NEXT:    %[[YSL:.+]] = impulse.slice %[[TR]] <start_indices = [0, 3], limit_indices = [1, 6], strides = [1, 1]> : (tensor<1x10xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:    %[[Y:.+]] = impulse.reshape %[[YSL]] : (tensor<1x3xf64>) -> tensor<3xf64>
// CHECK-NEXT:    %[[SCYSL:.+]] = impulse.slice %[[TR]] <start_indices = [0, 9], limit_indices = [1, 10], strides = [1, 1]> : (tensor<1x10xf64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[SCY:.+]] = impulse.reshape %[[SCYSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[LPY:.+]] = call @y_logpdf(%[[Y]], %[[MU]]) : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[DY:.+]] = arith.subf %[[LPY]], %[[SCY]] : tensor<f64>
// CHECK-NEXT:    %[[DELTA:.+]] = arith.addf %[[D1]], %[[DY]] : tensor<f64>
// CHECK:         %[[DFL:.+]] = impulse.reshape %[[DELTA]] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:    %[[P0:.+]] = impulse.dynamic_update_slice %{{.+}}, %[[DFL]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT:    %[[CFL:.+]] = impulse.reshape %[[CORR]] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:    %{{.+}} = impulse.dynamic_update_slice %[[P0]], %[[CFL]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
