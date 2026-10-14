// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_reshape_block_align" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A slice of whole blocks of a merged dimension, where the root is also
// reshaped with that dimension split into (blocks, size): the slice reads
// the blocked reshape.
func.func @aligned(%a: tensor<600xf64>) -> (tensor<8x25xf64>, tensor<8x3x25xf64>) {
  %m = stablehlo.reshape %a : (tensor<600xf64>) -> tensor<8x75xf64>
  %s = stablehlo.slice %m [0:8, 25:50] : (tensor<8x75xf64>) -> tensor<8x25xf64>
  %b = stablehlo.reshape %a : (tensor<600xf64>) -> tensor<8x3x25xf64>
  return %s, %b : tensor<8x25xf64>, tensor<8x3x25xf64>
}

// CHECK-LABEL: func.func @aligned(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<600xf64>) -> (tensor<8x25xf64>, tensor<8x3x25xf64>) {
// CHECK-NEXT: %[[B:[^ ,:)]+]] = stablehlo.reshape %[[A]] : (tensor<600xf64>) -> tensor<8x3x25xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.slice %[[B]] [0:8, 1:2, 0:25] : (tensor<8x3x25xf64>) -> tensor<8x1x25xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reshape %[[S]] : (tensor<8x1x25xf64>) -> tensor<8x25xf64>
// CHECK-NEXT: return %[[R]], %[[B]] : tensor<8x25xf64>, tensor<8x3x25xf64>
// CHECK-NEXT: }

// Not whole blocks: left alone.
func.func @unaligned(%a: tensor<600xf64>) -> (tensor<8x30xf64>, tensor<8x3x25xf64>) {
  %m = stablehlo.reshape %a : (tensor<600xf64>) -> tensor<8x75xf64>
  %s = stablehlo.slice %m [0:8, 20:50] : (tensor<8x75xf64>) -> tensor<8x30xf64>
  %b = stablehlo.reshape %a : (tensor<600xf64>) -> tensor<8x3x25xf64>
  return %s, %b : tensor<8x30xf64>, tensor<8x3x25xf64>
}

// CHECK-LABEL: func.func @unaligned(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<600xf64>) -> (tensor<8x30xf64>, tensor<8x3x25xf64>) {
// CHECK-NEXT: %[[M:[^ ,:)]+]] = stablehlo.reshape %[[A]] : (tensor<600xf64>) -> tensor<8x75xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.slice %[[M]] [0:8, 20:50] : (tensor<8x75xf64>) -> tensor<8x30xf64>
// CHECK-NEXT: %[[B:[^ ,:)]+]] = stablehlo.reshape %[[A]] : (tensor<600xf64>) -> tensor<8x3x25xf64>
// CHECK-NEXT: return %[[S]], %[[B]] : tensor<8x30xf64>, tensor<8x3x25xf64>
// CHECK-NEXT: }
