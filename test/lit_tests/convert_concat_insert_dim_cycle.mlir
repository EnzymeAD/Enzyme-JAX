// RUN: enzymexlamlir-opt --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// `convert_concat`, `elementwise_reshape_like` and `concat_insert_dim_elementwise`
// used to form a rewrite cycle on `convert(concat(reshape_i(x_i)))`:
//   convert_concat                 -> concat(convert(reshape(x_i)))
//   elementwise_reshape_like       -> concat(reshape(convert(x_i)))
//   concat_insert_dim_elementwise  -> convert(concat(reshape(x_i)))   (start)
// and every trip through the batching rewrite left another unbatched wrapper
// function behind, so the greedy driver never reached a fixed point. The convert
// is not pushed into the batchable form any more; this module is already at its
// fixed point and must come back unchanged.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.enzyme_hlo.concat_insert_dim_elementwise
      transform.apply_patterns.enzyme_hlo.elementwise_reshape_like
      transform.apply_patterns.enzyme_hlo.convert_concat
    } : !transform.any_op
    transform.yield
  }
  func.func @cycle(%arg0: tensor<16x64x2xf32>, %arg1: tensor<16x64x2xf32>) -> tensor<2x16x64x2xbf16> {
    %0 = stablehlo.reshape %arg0 : (tensor<16x64x2xf32>) -> tensor<1x16x64x2xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<16x64x2xf32>) -> tensor<1x16x64x2xf32>
    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1x16x64x2xf32>, tensor<1x16x64x2xf32>) -> tensor<2x16x64x2xf32>
    %3 = stablehlo.convert %2 : (tensor<2x16x64x2xf32>) -> tensor<2x16x64x2xbf16>
    return %3 : tensor<2x16x64x2xbf16>
  }
}

// CHECK:      func.func @cycle(%arg0: tensor<16x64x2xf32>, %arg1: tensor<16x64x2xf32>) -> tensor<2x16x64x2xbf16> {
// CHECK-NEXT:   %[[R0:.+]] = stablehlo.reshape %arg0 : (tensor<16x64x2xf32>) -> tensor<1x16x64x2xf32>
// CHECK-NEXT:   %[[R1:.+]] = stablehlo.reshape %arg1 : (tensor<16x64x2xf32>) -> tensor<1x16x64x2xf32>
// CHECK-NEXT:   %[[C:.+]] = stablehlo.concatenate %[[R0]], %[[R1]], dim = 0 : (tensor<1x16x64x2xf32>, tensor<1x16x64x2xf32>) -> tensor<2x16x64x2xf32>
// CHECK-NEXT:   %[[CV:.+]] = stablehlo.convert %[[C]] : (tensor<2x16x64x2xf32>) -> tensor<2x16x64x2xbf16>
// CHECK-NEXT:   return %[[CV]] : tensor<2x16x64x2xbf16>
// CHECK-NEXT: }
