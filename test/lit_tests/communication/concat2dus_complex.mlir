// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{concat_to_dus=1})" %s | FileCheck %s

sdy.mesh @mesh = <["x"=4]>

func.func @main(%arg0: tensor<48xcomplex<f64>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}, %arg1: tensor<192xcomplex<f64>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<192xcomplex<f64>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.slice %arg1 [48:192] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<192xcomplex<f64>>) -> tensor<144xcomplex<f64>>
  %1 = stablehlo.concatenate %arg0, %0, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<48xcomplex<f64>>, tensor<144xcomplex<f64>>) -> tensor<192xcomplex<f64>>
  return %1 : tensor<192xcomplex<f64>>
}

// CHECK-LABEL: func.func @main
// CHECK: %[[ZERO:.+]] = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>
// CHECK: stablehlo.pad %arg0, %[[ZERO]], low = [0], high = [144], interior = [0]
// CHECK: stablehlo.pad %{{.+}}, %[[ZERO]], low = [48], high = [0], interior = [0]
