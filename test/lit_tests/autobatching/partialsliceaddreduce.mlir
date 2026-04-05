// RUN: enzymexlamlir-opt --transform-interpreter %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.enzyme_hlo.add_reduce_slice_fusion
    } : !transform.any_op
    transform.yield
  }
  func.func @main(%arg0: tensor<40x104x56xf64>) -> tensor<96x48xf64> {
    %95 = stablehlo.slice %arg0 [35:36, 4:100, 4:52] : (tensor<40x104x56xf64>) -> tensor<1x96x48xf64>
    %96 = stablehlo.reshape %95 : (tensor<1x96x48xf64>) -> tensor<96x48xf64>
    %97 = stablehlo.slice %arg0 [36:37, 4:100, 4:52] : (tensor<40x104x56xf64>) -> tensor<1x96x48xf64>
    %98 = stablehlo.reshape %97 : (tensor<1x96x48xf64>) -> tensor<96x48xf64>
    %99 = stablehlo.add %96, %98 : tensor<96x48xf64>
    return %99 : tensor<96x48xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<40x104x56xf64>) -> tensor<96x48xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [35:37, 4:100, 4:52] : (tensor<40x104x56xf64>) -> tensor<2x96x48xf64>
// CHECK-NEXT:     %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2x96x48xf64>, tensor<f64>) -> tensor<96x48xf64>
// CHECK-NEXT:     return %1 : tensor<96x48xf64>
// CHECK-NEXT: }
