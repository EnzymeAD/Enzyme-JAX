// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=reshape_pad" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A per-element scratch row of 5x5 values, flattened, grown by 25 and viewed
// again as rows of 5: the pad adds 5 whole rows.
func.func @split_padded_dim(%a: tensor<8x25xf64>) -> tensor<8x10x5xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %p = stablehlo.pad %a, %z, low = [0, 0], high = [0, 25], interior = [0, 0] : (tensor<8x25xf64>, tensor<f64>) -> tensor<8x50xf64>
  %r = stablehlo.reshape %p : (tensor<8x50xf64>) -> tensor<8x10x5xf64>
  return %r : tensor<8x10x5xf64>
}

// CHECK-LABEL: func.func @split_padded_dim(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x25xf64>) -> tensor<8x10x5xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<8x25xf64>) -> tensor<8x5x5xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.pad %[[V2]], %[[V1]], low = [0, 0, 0], high = [0, 5, 0], interior = [0, 0, 0] : (tensor<8x5x5xf64>, tensor<f64>) -> tensor<8x10x5xf64>
// CHECK-NEXT: return %[[V3]] : tensor<8x10x5xf64>
// CHECK-NEXT: }

// Padding the outer of two merged dims pads the merged dim by whole rows.
func.func @merge_outer_padded(%a: tensor<8x5x5xf64>) -> tensor<8x1x35xf64> {
  %z = stablehlo.constant dense<1.5> : tensor<f64>
  %p = stablehlo.pad %a, %z, low = [0, 2, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<8x5x5xf64>, tensor<f64>) -> tensor<8x7x5xf64>
  %r = stablehlo.reshape %p : (tensor<8x7x5xf64>) -> tensor<8x1x35xf64>
  return %r : tensor<8x1x35xf64>
}

// CHECK-LABEL: func.func @merge_outer_padded(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x5x5xf64>) -> tensor<8x1x35xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<1.500000e+00> : tensor<f64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<8x5x5xf64>) -> tensor<8x1x25xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.pad %[[V2]], %[[V1]], low = [0, 0, 10], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<8x1x25xf64>, tensor<f64>) -> tensor<8x1x35xf64>
// CHECK-NEXT: return %[[V3]] : tensor<8x1x35xf64>
// CHECK-NEXT: }

// Cropping by a negative pad moves past the reshape the same way.
func.func @split_negative(%a: tensor<8x30xf64>) -> tensor<8x5x5xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %p = stablehlo.pad %a, %z, low = [0, -5], high = [0, 0], interior = [0, 0] : (tensor<8x30xf64>, tensor<f64>) -> tensor<8x25xf64>
  %r = stablehlo.reshape %p : (tensor<8x25xf64>) -> tensor<8x5x5xf64>
  return %r : tensor<8x5x5xf64>
}

// CHECK-LABEL: func.func @split_negative(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x30xf64>) -> tensor<8x5x5xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<8x30xf64>) -> tensor<8x6x5xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.pad %[[V2]], %[[V1]], low = [0, -1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<8x6x5xf64>, tensor<f64>) -> tensor<8x5x5xf64>
// CHECK-NEXT: return %[[V3]] : tensor<8x5x5xf64>
// CHECK-NEXT: }

// Three elements are not a whole row of 4: the pad stays.
func.func @split_partial_row(%a: tensor<8x25xf64>) -> tensor<8x7x4xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %p = stablehlo.pad %a, %z, low = [0, 0], high = [0, 3], interior = [0, 0] : (tensor<8x25xf64>, tensor<f64>) -> tensor<8x28xf64>
  %r = stablehlo.reshape %p : (tensor<8x28xf64>) -> tensor<8x7x4xf64>
  return %r : tensor<8x7x4xf64>
}

// CHECK-LABEL: func.func @split_partial_row(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x25xf64>) -> tensor<8x7x4xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.pad %[[V0]], %[[V1]], low = [0, 0], high = [0, 3], interior = [0, 0] : (tensor<8x25xf64>, tensor<f64>) -> tensor<8x28xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<8x28xf64>) -> tensor<8x7x4xf64>
// CHECK-NEXT: return %[[V3]] : tensor<8x7x4xf64>
// CHECK-NEXT: }

// Padding the inner of two merged dims interleaves the padding: the pad stays.
func.func @merge_inner_padded(%a: tensor<8x5x5xf64>) -> tensor<8x30xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %p = stablehlo.pad %a, %z, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<8x5x5xf64>, tensor<f64>) -> tensor<8x5x6xf64>
  %r = stablehlo.reshape %p : (tensor<8x5x6xf64>) -> tensor<8x30xf64>
  return %r : tensor<8x30xf64>
}

// CHECK-LABEL: func.func @merge_inner_padded(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x5x5xf64>) -> tensor<8x30xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.pad %[[V0]], %[[V1]], low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<8x5x5xf64>, tensor<f64>) -> tensor<8x5x6xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<8x5x6xf64>) -> tensor<8x30xf64>
// CHECK-NEXT: return %[[V3]] : tensor<8x30xf64>
// CHECK-NEXT: }
