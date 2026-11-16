// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=add_terms_to_reduce max-iterations=0" %s | FileCheck %s --check-prefix=GEN
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=add_terms_to_reduce max-iterations=0" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=APPLY

// With max-iterations=0 the generated apply_patterns op carries the limit and
// the greedy driver is not allowed to iterate, so the pattern is not applied.

// GEN-LABEL: transform.named_sequence @__transform_main
// GEN: transform.apply_patterns to %{{.*}} {
// GEN-NEXT: transform.apply_patterns.enzyme_hlo.add_terms_to_reduce
// GEN: } max_iterations = 0 : !transform.any_op
// GEN-NOT: max_num_rewrites

// APPLY-LABEL: func.func @interleaved(
// APPLY-NOT: stablehlo.reduce
func.func @interleaved(%x: tensor<8x9xf64>, %b: tensor<9xf64>) -> tensor<8x3xf64> {
  %x0 = stablehlo.slice %x [0:8, 0:9:3] : (tensor<8x9xf64>) -> tensor<8x3xf64>
  %x1 = stablehlo.slice %x [0:8, 1:9:3] : (tensor<8x9xf64>) -> tensor<8x3xf64>
  %x2 = stablehlo.slice %x [0:8, 2:9:3] : (tensor<8x9xf64>) -> tensor<8x3xf64>
  %b0 = stablehlo.slice %b [0:3] : (tensor<9xf64>) -> tensor<3xf64>
  %b1 = stablehlo.slice %b [3:6] : (tensor<9xf64>) -> tensor<3xf64>
  %b2 = stablehlo.slice %b [6:9] : (tensor<9xf64>) -> tensor<3xf64>
  %c0 = stablehlo.broadcast_in_dim %b0, dims = [1] : (tensor<3xf64>) -> tensor<8x3xf64>
  %c1 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<3xf64>) -> tensor<8x3xf64>
  %c2 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<3xf64>) -> tensor<8x3xf64>
  %t0 = stablehlo.multiply %x0, %c0 : tensor<8x3xf64>
  %t1 = stablehlo.multiply %x1, %c1 : tensor<8x3xf64>
  %t2 = stablehlo.multiply %x2, %c2 : tensor<8x3xf64>
  %s0 = stablehlo.add %t0, %t1 : tensor<8x3xf64>
  %s1 = stablehlo.add %s0, %t2 : tensor<8x3xf64>
  return %s1 : tensor<8x3xf64>
}
