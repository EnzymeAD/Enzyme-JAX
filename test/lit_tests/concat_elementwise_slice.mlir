// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=concat_elementwise_slice" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A concatenate of slices of the same elementwise op over different
// operands becomes the op over a concatenate of the sliced operands, even
// though the op has other users: the slices then reach the arguments and
// the concatenate is one of views. The constant operand is one splat
// again, not a concatenate.

// CHECK-LABEL: @slices_of_multiplies
// CHECK-DAG: %[[A0:.+]] = stablehlo.slice %arg0 [0:3, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG: %[[B1:.+]] = stablehlo.slice %arg1 [3:7, 0:4] : (tensor<8x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG: %[[A2:.+]] = stablehlo.slice %arg0 [7:8, 0:4] : (tensor<8x4xf32>) -> tensor<1x4xf32>
// CHECK-DAG: %[[X:.+]] = stablehlo.concatenate %[[A0]], %[[B1]], %[[A2]], dim = 0 : (tensor<3x4xf32>, tensor<4x4xf32>, tensor<1x4xf32>) -> tensor<8x4xf32>
// CHECK-DAG: %[[C:.+]] = stablehlo.constant dense<1.500000e+00> : tensor<8x4xf32>
// CHECK-DAG: %[[MA:.+]] = stablehlo.multiply %arg0, %{{.+}} : tensor<8x4xf32>
// CHECK: %[[R:.+]] = stablehlo.multiply %[[X]], %[[C]] : tensor<8x4xf32>
// CHECK: return %[[R]], %[[MA]]
func.func @slices_of_multiplies(%a: tensor<8x4xf32>, %b: tensor<8x4xf32>) -> (tensor<8x4xf32>, tensor<8x4xf32>) {
  %cst = stablehlo.constant dense<1.5> : tensor<8x4xf32>
  %ma = stablehlo.multiply %a, %cst : tensor<8x4xf32>
  %mb = stablehlo.multiply %b, %cst : tensor<8x4xf32>
  %p0 = stablehlo.slice %ma [0:3, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
  %p1 = stablehlo.slice %mb [3:7, 0:4] : (tensor<8x4xf32>) -> tensor<4x4xf32>
  %p2 = stablehlo.slice %ma [7:8, 0:4] : (tensor<8x4xf32>) -> tensor<1x4xf32>
  %r = stablehlo.concatenate %p0, %p1, %p2, dim = 0 : (tensor<3x4xf32>, tensor<4x4xf32>, tensor<1x4xf32>) -> tensor<8x4xf32>
  return %r, %ma : tensor<8x4xf32>, tensor<8x4xf32>
}

// -----

// A piece that is the op itself (not a slice of it) joins with its operands
// as they are. A unary op needs one concatenate whatever its operands are.

// CHECK-LABEL: @mixed_pieces
// CHECK-DAG: %[[A0:.+]] = stablehlo.slice %arg0 [0:3, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG: %[[X:.+]] = stablehlo.concatenate %[[A0]], %arg1, dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
// CHECK: %[[R:.+]] = stablehlo.negate %[[X]] : tensor<8x4xf32>
// CHECK: return %[[R]]
func.func @mixed_pieces(%a: tensor<8x4xf32>, %b: tensor<5x4xf32>) -> tensor<8x4xf32> {
  %na = stablehlo.negate %a : tensor<8x4xf32>
  %nb = stablehlo.negate %b : tensor<5x4xf32>
  %p0 = stablehlo.slice %na [0:3, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
  %r = stablehlo.concatenate %p0, %nb, dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}

// -----

// A computed operand the pieces read at adjacent windows of one value is
// that value (or a slice of it), not a concatenate: the hoist is made and
// the concatenate moves down the chain of the other operand, here through
// the unary exponential to the arguments.

// CHECK-LABEL: @shared_computed_operand
// CHECK-DAG: %[[T:.+]] = stablehlo.tanh %arg0 : tensor<8x4xf32>
// CHECK-DAG: %[[A0:.+]] = stablehlo.slice %arg0 [0:3, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG: %[[B1:.+]] = stablehlo.slice %arg1 [3:8, 0:4] : (tensor<8x4xf32>) -> tensor<5x4xf32>
// CHECK-DAG: %[[X:.+]] = stablehlo.concatenate %[[A0]], %[[B1]], dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
// CHECK-DAG: %[[E:.+]] = stablehlo.exponential %[[X]] : tensor<8x4xf32>
// CHECK: %[[R:.+]] = stablehlo.multiply %[[E]], %[[T]] : tensor<8x4xf32>
// CHECK: return %[[R]]
func.func @shared_computed_operand(%a: tensor<8x4xf32>, %b: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %t = stablehlo.tanh %a : tensor<8x4xf32>
  %ea = stablehlo.exponential %a : tensor<8x4xf32>
  %eb = stablehlo.exponential %b : tensor<8x4xf32>
  %ma = stablehlo.multiply %ea, %t : tensor<8x4xf32>
  %mb = stablehlo.multiply %eb, %t : tensor<8x4xf32>
  %p0 = stablehlo.slice %ma [0:3, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
  %p1 = stablehlo.slice %mb [3:8, 0:4] : (tensor<8x4xf32>) -> tensor<5x4xf32>
  %r = stablehlo.concatenate %p0, %p1, dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}

// -----

// Broadcasts of the same scalar behind every piece become one broadcast.

// CHECK-LABEL: @shared_scalar
// CHECK-DAG: %[[S:.+]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<8x4xf32>
// CHECK-DAG: %[[A0:.+]] = stablehlo.slice %arg0 [0:3, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG: %[[X:.+]] = stablehlo.concatenate %[[A0]], %arg1, dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
// CHECK: %[[R:.+]] = stablehlo.multiply %[[X]], %[[S]] : tensor<8x4xf32>
// CHECK: return %[[R]]
func.func @shared_scalar(%a: tensor<8x4xf32>, %b: tensor<5x4xf32>, %s: tensor<f32>) -> tensor<8x4xf32> {
  %s8 = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f32>) -> tensor<8x4xf32>
  %s5 = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f32>) -> tensor<5x4xf32>
  %ma = stablehlo.multiply %a, %s8 : tensor<8x4xf32>
  %mb = stablehlo.multiply %b, %s5 : tensor<5x4xf32>
  %p0 = stablehlo.slice %ma [0:3, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
  %r = stablehlo.concatenate %p0, %mb, dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}

// -----

// Two operand positions would each need a concatenate: left alone.

// CHECK-LABEL: @two_concatenates
// CHECK: stablehlo.concatenate
// CHECK-NOT: stablehlo.concatenate
func.func @two_concatenates(%a: tensor<8x4xf32>, %b: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %ea = stablehlo.exponential %a : tensor<8x4xf32>
  %eb = stablehlo.exponential %b : tensor<8x4xf32>
  %ma = stablehlo.multiply %ea, %a : tensor<8x4xf32>
  %mb = stablehlo.multiply %eb, %b : tensor<8x4xf32>
  %p0 = stablehlo.slice %ma [0:3, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
  %p1 = stablehlo.slice %mb [3:8, 0:4] : (tensor<8x4xf32>) -> tensor<5x4xf32>
  %r = stablehlo.concatenate %p0, %p1, dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}

// -----

// Different ops behind the pieces: left alone.

// CHECK-LABEL: @different_ops
// CHECK: stablehlo.concatenate
func.func @different_ops(%a: tensor<8x4xf32>, %b: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %na = stablehlo.negate %a : tensor<8x4xf32>
  %ab = stablehlo.abs %b : tensor<8x4xf32>
  %p0 = stablehlo.slice %na [0:3, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
  %p1 = stablehlo.slice %ab [3:8, 0:4] : (tensor<8x4xf32>) -> tensor<5x4xf32>
  %r = stablehlo.concatenate %p0, %p1, dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}

// -----

// The same op on operands of different element types (a compare of f32
// and one of f64, both i1): the operand concatenate would not type.

// CHECK-LABEL: @operand_types_differ
// CHECK: stablehlo.concatenate
func.func @operand_types_differ(%a: tensor<8x4xf32>, %b: tensor<8x4xf64>) -> tensor<8x4xi1> {
  %za = stablehlo.constant dense<0.0> : tensor<8x4xf32>
  %zb = stablehlo.constant dense<0.0> : tensor<8x4xf64>
  %ca = stablehlo.compare LT, %a, %za : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xi1>
  %cb = stablehlo.compare LT, %b, %zb : (tensor<8x4xf64>, tensor<8x4xf64>) -> tensor<8x4xi1>
  %p0 = stablehlo.slice %ca [0:3, 0:4] : (tensor<8x4xi1>) -> tensor<3x4xi1>
  %p1 = stablehlo.slice %cb [3:8, 0:4] : (tensor<8x4xi1>) -> tensor<5x4xi1>
  %r = stablehlo.concatenate %p0, %p1, dim = 0 : (tensor<3x4xi1>, tensor<5x4xi1>) -> tensor<8x4xi1>
  return %r : tensor<8x4xi1>
}

// -----

// No piece is a slice: ConcatElementwise's case, not this one.

// CHECK-LABEL: @no_slice
// CHECK: stablehlo.concatenate
func.func @no_slice(%a: tensor<3x4xf32>, %b: tensor<5x4xf32>) -> tensor<8x4xf32> {
  %na = stablehlo.negate %a : tensor<3x4xf32>
  %nb = stablehlo.negate %b : tensor<5x4xf32>
  %r = stablehlo.concatenate %na, %nb, dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}

// -----

// A slice with another user survives the rewrite: left alone.

// CHECK-LABEL: @slice_used_elsewhere
// CHECK: stablehlo.concatenate
func.func @slice_used_elsewhere(%a: tensor<8x4xf32>, %b: tensor<8x4xf32>) -> (tensor<8x4xf32>, tensor<3x4xf32>) {
  %na = stablehlo.negate %a : tensor<8x4xf32>
  %nb = stablehlo.negate %b : tensor<8x4xf32>
  %p0 = stablehlo.slice %na [0:3, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
  %p1 = stablehlo.slice %nb [3:8, 0:4] : (tensor<8x4xf32>) -> tensor<5x4xf32>
  %r = stablehlo.concatenate %p0, %p1, dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
  return %r, %p0 : tensor<8x4xf32>, tensor<3x4xf32>
}
