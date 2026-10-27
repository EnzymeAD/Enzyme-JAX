// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=concat_elementwise_slice" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A concatenate of slices of the same elementwise op over different view
// operands becomes the op over concatenates of the sliced operands, even
// though the op has other users: the slices then reach the arguments and
// the concatenate is one of views.

// CHECK-LABEL: @slices_of_multiplies
// CHECK-DAG: %[[A0:.+]] = stablehlo.slice %arg0 [0:3, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG: %[[B1:.+]] = stablehlo.slice %arg1 [3:7, 0:4] : (tensor<8x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG: %[[A2:.+]] = stablehlo.slice %arg0 [7:8, 0:4] : (tensor<8x4xf32>) -> tensor<1x4xf32>
// CHECK-DAG: %[[X:.+]] = stablehlo.concatenate %[[A0]], %[[B1]], %[[A2]], dim = 0 : (tensor<3x4xf32>, tensor<4x4xf32>, tensor<1x4xf32>) -> tensor<8x4xf32>
// CHECK-DAG: %[[C0:.+]] = stablehlo.slice %[[CST:.+]] [0:3, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG: %[[C1:.+]] = stablehlo.slice %[[CST]] [3:7, 0:4] : (tensor<8x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG: %[[C2:.+]] = stablehlo.slice %[[CST]] [7:8, 0:4] : (tensor<8x4xf32>) -> tensor<1x4xf32>
// CHECK-DAG: %[[C:.+]] = stablehlo.concatenate %[[C0]], %[[C1]], %[[C2]], dim = 0
// CHECK-DAG: %[[MA:.+]] = stablehlo.multiply %arg0, %[[CST]] : tensor<8x4xf32>
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
// as they are.

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
// and one of f64, both i1): the operand concatenates would not type.

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

// The op's operands are computed: hoisting would leave a concatenate per
// operand of the next level too. Left alone.

// CHECK-LABEL: @computed_operands
// CHECK: stablehlo.concatenate
func.func @computed_operands(%a: tensor<8x4xf32>, %b: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %ea = stablehlo.exponential %a : tensor<8x4xf32>
  %eb = stablehlo.exponential %b : tensor<8x4xf32>
  %ma = stablehlo.multiply %ea, %a : tensor<8x4xf32>
  %mb = stablehlo.multiply %eb, %b : tensor<8x4xf32>
  %p0 = stablehlo.slice %ma [0:3, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
  %p1 = stablehlo.slice %mb [3:8, 0:4] : (tensor<8x4xf32>) -> tensor<5x4xf32>
  %r = stablehlo.concatenate %p0, %p1, dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}
