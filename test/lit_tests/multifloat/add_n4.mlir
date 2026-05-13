// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=first" %s | FileCheck %s --check-prefix=FIRST
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=last" %s | FileCheck %s --check-prefix=LAST
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @do_add(%a: tensor<4xf64>, %b: tensor<4xf64>) -> tensor<4xf64> {
  %r = stablehlo.add %a, %b : tensor<4xf64>
  return %r : tensor<4xf64>
}

// FIRST-LABEL: func.func @do_add
// FIRST: stablehlo.concatenate {{.*}} dim = 0 : (tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<4x4xf32>
// FIRST: stablehlo.add %{{.*}} : tensor<1x4xf32>

// LAST-LABEL: func.func @do_add
// LAST: stablehlo.concatenate {{.*}} : (tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x4xf32>
// LAST: stablehlo.add %{{.*}} : tensor<4x1xf32>

func.func @main() attributes {enzyme.no_multifloat} {
  %a = stablehlo.constant dense<[1.0, 0.1, 1.0e15, 2.0]> : tensor<4xf64>
  %b = stablehlo.constant dense<[2.0, 0.2, 1.0, -2.0]> : tensor<4xf64>
  %expected = stablehlo.constant dense<[3.0, 0.30000000000000004, 1.000000000000001e15, 0.0]> : tensor<4xf64>

  %r = func.call @do_add(%a, %b) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  "check.expect_close"(%r, %expected) {max_ulp_difference = 4 : ui64} : (tensor<4xf64>, tensor<4xf64>) -> ()

  return
}
