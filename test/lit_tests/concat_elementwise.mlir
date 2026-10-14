// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_elementwise" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Analysis annotations (enzymexla.*) describe an op's values, not what it
// computes: adds that differ only in them are the same op, concatenated
// through, and the add of the concatenations does not carry them.
func.func @annotated(%a: tensor<8x3x3xf64>, %b: tensor<8x3x3xf64>, %c: tensor<8x3x3xf64>, %d: tensor<8x3x3xf64>) -> tensor<8x6x3xf64> {
  %0 = stablehlo.add %a, %b {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<8x3x3xf64>
  %1 = stablehlo.add %c, %d : tensor<8x3x3xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<8x3x3xf64>, tensor<8x3x3xf64>) -> tensor<8x6x3xf64>
  return %2 : tensor<8x6x3xf64>
}

// CHECK-LABEL: func.func @annotated(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<8x3x3xf64>, %[[B:[^ ,:)]+]]: tensor<8x3x3xf64>, %[[C:[^ ,:)]+]]: tensor<8x3x3xf64>, %[[D:[^ ,:)]+]]: tensor<8x3x3xf64>) -> tensor<8x6x3xf64> {
// CHECK-NEXT: %[[L:[^ ,:)]+]] = stablehlo.concatenate %[[A]], %[[C]], dim = 1 : (tensor<8x3x3xf64>, tensor<8x3x3xf64>) -> tensor<8x6x3xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.concatenate %[[B]], %[[D]], dim = 1 : (tensor<8x3x3xf64>, tensor<8x3x3xf64>) -> tensor<8x6x3xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.add %[[L]], %[[R]] : tensor<8x6x3xf64>
// CHECK-NEXT: return %[[S]] : tensor<8x6x3xf64>
// CHECK-NEXT: }

// Ops that differ in an attribute that defines the computation stay apart.
func.func @compare(%a: tensor<8x3xf64>, %b: tensor<8x3xf64>) -> tensor<8x6xi1> {
  %0 = stablehlo.compare LT, %a, %b : (tensor<8x3xf64>, tensor<8x3xf64>) -> tensor<8x3xi1>
  %1 = stablehlo.compare GT, %a, %b : (tensor<8x3xf64>, tensor<8x3xf64>) -> tensor<8x3xi1>
  %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<8x3xi1>, tensor<8x3xi1>) -> tensor<8x6xi1>
  return %2 : tensor<8x6xi1>
}

// CHECK-LABEL: func.func @compare(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<8x3xf64>, %[[B:[^ ,:)]+]]: tensor<8x3xf64>) -> tensor<8x6xi1> {
// CHECK-NEXT: %[[LT:[^ ,:)]+]] = stablehlo.compare LT, %[[A]], %[[B]] : (tensor<8x3xf64>, tensor<8x3xf64>) -> tensor<8x3xi1>
// CHECK-NEXT: %[[GT:[^ ,:)]+]] = stablehlo.compare GT, %[[A]], %[[B]] : (tensor<8x3xf64>, tensor<8x3xf64>) -> tensor<8x3xi1>
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.concatenate %[[LT]], %[[GT]], dim = 1 : (tensor<8x3xi1>, tensor<8x3xi1>) -> tensor<8x6xi1>
// CHECK-NEXT: return %[[C]] : tensor<8x6xi1>
// CHECK-NEXT: }
