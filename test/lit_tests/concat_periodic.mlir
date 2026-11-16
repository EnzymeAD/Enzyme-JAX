// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_periodic" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Operands repeating with a period (the basis rows a kernel selects per
// direction, [B; G] three times): the period concatenated once, repeated
// along a new major dimension by a broadcast, and the two merged.
func.func @periodic(%a: tensor<5x5x5xf64>, %b: tensor<5x20x5xf64>) -> tensor<5x75x5xf64> {
  %0 = stablehlo.concatenate %a, %b, %a, %b, %a, %b, dim = 1 : (tensor<5x5x5xf64>, tensor<5x20x5xf64>, tensor<5x5x5xf64>, tensor<5x20x5xf64>, tensor<5x5x5xf64>, tensor<5x20x5xf64>) -> tensor<5x75x5xf64>
  return %0 : tensor<5x75x5xf64>
}

// CHECK-LABEL: func.func @periodic(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<5x5x5xf64>, %[[B:[^ ,:)]+]]: tensor<5x20x5xf64>) -> tensor<5x75x5xf64> {
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.concatenate %[[A]], %[[B]], dim = 1 : (tensor<5x5x5xf64>, tensor<5x20x5xf64>) -> tensor<5x25x5xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[C]], dims = [0, 2, 3] : (tensor<5x25x5xf64>) -> tensor<5x3x25x5xf64>
// CHECK-NEXT: %[[M:[^ ,:)]+]] = stablehlo.reshape %[[R]] : (tensor<5x3x25x5xf64>) -> tensor<5x75x5xf64>
// CHECK-NEXT: return %[[M]] : tensor<5x75x5xf64>
// CHECK-NEXT: }

// The period must divide the operand count: a, b, a, b, a is not periodic.
func.func @not_periodic(%a: tensor<5x5x5xf64>, %b: tensor<5x20x5xf64>) -> tensor<5x55x5xf64> {
  %0 = stablehlo.concatenate %a, %b, %a, %b, %a, dim = 1 : (tensor<5x5x5xf64>, tensor<5x20x5xf64>, tensor<5x5x5xf64>, tensor<5x20x5xf64>, tensor<5x5x5xf64>) -> tensor<5x55x5xf64>
  return %0 : tensor<5x55x5xf64>
}

// CHECK-LABEL: func.func @not_periodic(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<5x5x5xf64>, %[[B:[^ ,:)]+]]: tensor<5x20x5xf64>) -> tensor<5x55x5xf64> {
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.concatenate %[[A]], %[[B]], %[[A]], %[[B]], %[[A]], dim = 1 : (tensor<5x5x5xf64>, tensor<5x20x5xf64>, tensor<5x5x5xf64>, tensor<5x20x5xf64>, tensor<5x5x5xf64>) -> tensor<5x55x5xf64>
// CHECK-NEXT: return %[[C]] : tensor<5x55x5xf64>
// CHECK-NEXT: }

// All operands the same is concat_to_broadcast's (period one).
func.func @all_same(%a: tensor<5x5x5xf64>) -> tensor<5x15x5xf64> {
  %0 = stablehlo.concatenate %a, %a, %a, dim = 1 : (tensor<5x5x5xf64>, tensor<5x5x5xf64>, tensor<5x5x5xf64>) -> tensor<5x15x5xf64>
  return %0 : tensor<5x15x5xf64>
}

// CHECK-LABEL: func.func @all_same(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<5x5x5xf64>) -> tensor<5x15x5xf64> {
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.concatenate %[[A]], %[[A]], %[[A]], dim = 1 : (tensor<5x5x5xf64>, tensor<5x5x5xf64>, tensor<5x5x5xf64>) -> tensor<5x15x5xf64>
// CHECK-NEXT: return %[[C]] : tensor<5x15x5xf64>
// CHECK-NEXT: }
