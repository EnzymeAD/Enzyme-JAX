// See dot_precise_n3.mlir for full explanation. N=4 path.
//
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=first precise-reduce=true" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=last precise-reduce=true" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @do_dot4(%a: tensor<4xf64>, %b: tensor<4xf64>) -> tensor<f64> {
  %r = stablehlo.dot_general %a, %b, contracting_dims = [0] x [0] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
  return %r : tensor<f64>
}

func.func @do_dot3(%a: tensor<3xf64>, %b: tensor<3xf64>) -> tensor<f64> {
  %r = stablehlo.dot_general %a, %b, contracting_dims = [0] x [0] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
  return %r : tensor<f64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  %a1 = stablehlo.constant dense<[1.0e10, 1.0, 0.1, 0.01]> : tensor<4xf64>
  %b1 = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf64>
  %e1 = stablehlo.constant dense<10000000001.110001> : tensor<f64>
  %r1 = func.call @do_dot4(%a1, %b1) : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
  "check.expect_close"(%r1, %e1) {max_ulp_difference = 4 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  %a2 = stablehlo.constant dense<[1.0e30, 1.0, -1.0e30]> : tensor<3xf64>
  %b2 = stablehlo.constant dense<[1.0, 1.0, 1.0]> : tensor<3xf64>
  %e2 = stablehlo.constant dense<1.0> : tensor<f64>
  %r2 = func.call @do_dot3(%a2, %b2) : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
  "check.expect_close"(%r2, %e2) {max_ulp_difference = 4 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  return
}
