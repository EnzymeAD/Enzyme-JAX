// RUN: enzymexlamlir-opt --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A dot_general whose result element type differs from its operands' (a bf16 x
// bf16 -> f32 accumulating dot, what JAX emits with preferred_element_type=f32)
// must keep that result type when the feeding reshape is hoisted past it,
// otherwise the trailing reshape changes the element type and fails to verify.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.enzyme_hlo.dot_general_reshape
    } : !transform.any_op
    transform.yield
  }
  func.func @mixed(%arg0: tensor<4x8xbf16>, %arg1: tensor<8x16xbf16>) -> tensor<1x4x16xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<4x8xbf16>) -> tensor<1x4x8xbf16>
    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [2] x [0] : (tensor<1x4x8xbf16>, tensor<8x16xbf16>) -> tensor<1x4x16xf32>
    return %1 : tensor<1x4x16xf32>
  }
}

// CHECK:      func.func @mixed(%arg0: tensor<4x8xbf16>, %arg1: tensor<8x16xbf16>) -> tensor<1x4x16xf32> {
// CHECK-NEXT:   %[[DOT:.+]] = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]{{.*}} : (tensor<4x8xbf16>, tensor<8x16xbf16>) -> tensor<4x16xf32>
// CHECK-NEXT:   %[[RES:.+]] = stablehlo.reshape %[[DOT]] : (tensor<4x16xf32>) -> tensor<1x4x16xf32>
// CHECK-NEXT:   return %[[RES]] : tensor<1x4x16xf32>
// CHECK-NEXT: }
