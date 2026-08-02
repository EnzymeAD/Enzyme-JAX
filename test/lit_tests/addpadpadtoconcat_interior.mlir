// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=add_pad_pad_to_concat" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Must not fire: interior padding on dim 1 differs (0 vs 1).
// CHECK-LABEL: func.func @interior_mismatch
// CHECK-SAME:    (%[[A:.+]]: tensor<3x5xf32>, %[[B:.+]]: tensor<2x3xf32>) -> tensor<5x5xf32>
// CHECK-NEXT:    %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[P1:.+]] = stablehlo.pad %[[A]], %[[ZERO]], low = [2, 0], high = [0, 0], interior = [0, 0] : (tensor<3x5xf32>, tensor<f32>) -> tensor<5x5xf32>
// CHECK-NEXT:    %[[P2:.+]] = stablehlo.pad %[[B]], %[[ZERO]], low = [0, 0], high = [3, 0], interior = [0, 1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<5x5xf32>
// CHECK-NEXT:    %[[R:.+]] = stablehlo.add %[[P1]], %[[P2]] : tensor<5x5xf32>
// CHECK-NEXT:    return %[[R]] : tensor<5x5xf32>
func.func @interior_mismatch(%a: tensor<3x5xf32>, %b: tensor<2x3xf32>) -> tensor<5x5xf32> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %p1 = stablehlo.pad %a, %zero, low = [2, 0], high = [0, 0], interior = [0, 0] : (tensor<3x5xf32>, tensor<f32>) -> tensor<5x5xf32>
  %p2 = stablehlo.pad %b, %zero, low = [0, 0], high = [3, 0], interior = [0, 1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<5x5xf32>
  %r = stablehlo.add %p1, %p2 : tensor<5x5xf32>
  return %r : tensor<5x5xf32>
}

// CHECK-LABEL: func.func @interior_mismatch_swapped
// CHECK-SAME:    (%[[A:.+]]: tensor<3x5xf32>, %[[B:.+]]: tensor<2x3xf32>) -> tensor<5x5xf32>
// CHECK-NEXT:    %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[P1:.+]] = stablehlo.pad %[[B]], %[[ZERO]], low = [0, 0], high = [3, 0], interior = [0, 1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<5x5xf32>
// CHECK-NEXT:    %[[P2:.+]] = stablehlo.pad %[[A]], %[[ZERO]], low = [2, 0], high = [0, 0], interior = [0, 0] : (tensor<3x5xf32>, tensor<f32>) -> tensor<5x5xf32>
// CHECK-NEXT:    %[[R:.+]] = stablehlo.add %[[P1]], %[[P2]] : tensor<5x5xf32>
// CHECK-NEXT:    return %[[R]] : tensor<5x5xf32>
func.func @interior_mismatch_swapped(%a: tensor<3x5xf32>, %b: tensor<2x3xf32>) -> tensor<5x5xf32> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %p1 = stablehlo.pad %b, %zero, low = [0, 0], high = [3, 0], interior = [0, 1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<5x5xf32>
  %p2 = stablehlo.pad %a, %zero, low = [2, 0], high = [0, 0], interior = [0, 0] : (tensor<3x5xf32>, tensor<f32>) -> tensor<5x5xf32>
  %r = stablehlo.add %p1, %p2 : tensor<5x5xf32>
  return %r : tensor<5x5xf32>
}

// CHECK-LABEL: func.func @interior_match
// CHECK-SAME:    (%[[A:.+]]: tensor<3x3xf32>, %[[B:.+]]: tensor<2x3xf32>) -> tensor<5x5xf32>
// CHECK-NEXT:    %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[CAT:.+]] = stablehlo.concatenate %[[B]], %[[A]], dim = 0 : (tensor<2x3xf32>, tensor<3x3xf32>) -> tensor<5x3xf32>
// CHECK-NEXT:    %[[PAD:.+]] = stablehlo.pad %[[CAT]], %[[ZERO]], low = [0, 0], high = [0, 0], interior = [0, 1] : (tensor<5x3xf32>, tensor<f32>) -> tensor<5x5xf32>
// CHECK-NEXT:    return %[[PAD]] : tensor<5x5xf32>
func.func @interior_match(%a: tensor<3x3xf32>, %b: tensor<2x3xf32>) -> tensor<5x5xf32> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %p1 = stablehlo.pad %a, %zero, low = [2, 0], high = [0, 0], interior = [0, 1] : (tensor<3x3xf32>, tensor<f32>) -> tensor<5x5xf32>
  %p2 = stablehlo.pad %b, %zero, low = [0, 0], high = [3, 0], interior = [0, 1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<5x5xf32>
  %r = stablehlo.add %p1, %p2 : tensor<5x5xf32>
  return %r : tensor<5x5xf32>
}
