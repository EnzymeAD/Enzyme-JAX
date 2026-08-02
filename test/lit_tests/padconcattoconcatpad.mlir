// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=pad_concat_to_concat_pad" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Only the first operand's low padding may be hoisted on the concat dim.
// CHECK-LABEL: func.func @concat_dim_only
// CHECK-SAME:    (%[[X:.+]]: tensor<1x2xf64>, %[[Y:.+]]: tensor<1x2xf64>) -> tensor<1x6xf64>
// CHECK-NEXT:    %[[Z:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[B:.+]] = stablehlo.pad %[[Y]], %[[Z]], low = [0, 1], high = [0, 0], interior = [0, 0] : (tensor<1x2xf64>, tensor<f64>) -> tensor<1x3xf64>
// CHECK-NEXT:    %[[C:.+]] = stablehlo.concatenate %[[X]], %[[B]], dim = 1 : (tensor<1x2xf64>, tensor<1x3xf64>) -> tensor<1x5xf64>
// CHECK-NEXT:    %[[PAD:.+]] = stablehlo.pad %[[C]], %[[Z]], low = [0, 1], high = [0, 0], interior = [0, 0] : (tensor<1x5xf64>, tensor<f64>) -> tensor<1x6xf64>
// CHECK-NEXT:    return %[[PAD]] : tensor<1x6xf64>
func.func @concat_dim_only(%x: tensor<1x2xf64>, %y: tensor<1x2xf64>) -> tensor<1x6xf64> {
  %z = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %a = stablehlo.pad %x, %z, low = [0, 1], high = [0, 0], interior = [0, 0] : (tensor<1x2xf64>, tensor<f64>) -> tensor<1x3xf64>
  %b = stablehlo.pad %y, %z, low = [0, 1], high = [0, 0], interior = [0, 0] : (tensor<1x2xf64>, tensor<f64>) -> tensor<1x3xf64>
  %c = stablehlo.concatenate %a, %b, dim = 1 : (tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<1x6xf64>
  return %c : tensor<1x6xf64>
}

// CHECK-LABEL: func.func @non_concat_dim
// CHECK-SAME:    (%[[X:.+]]: tensor<1x2xf64>, %[[Y:.+]]: tensor<1x2xf64>) -> tensor<3x4xf64>
// CHECK-NEXT:    %[[Z:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[CAT:.+]] = stablehlo.concatenate %[[X]], %[[Y]], dim = 1 : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<1x4xf64>
// CHECK-NEXT:    %[[PAD:.+]] = stablehlo.pad %[[CAT]], %[[Z]], low = [1, 0], high = [1, 0], interior = [0, 0] : (tensor<1x4xf64>, tensor<f64>) -> tensor<3x4xf64>
// CHECK-NEXT:    return %[[PAD]] : tensor<3x4xf64>
func.func @non_concat_dim(%x: tensor<1x2xf64>, %y: tensor<1x2xf64>) -> tensor<3x4xf64> {
  %z = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %a = stablehlo.pad %x, %z, low = [1, 0], high = [1, 0], interior = [0, 0] : (tensor<1x2xf64>, tensor<f64>) -> tensor<3x2xf64>
  %b = stablehlo.pad %y, %z, low = [1, 0], high = [1, 0], interior = [0, 0] : (tensor<1x2xf64>, tensor<f64>) -> tensor<3x2xf64>
  %c = stablehlo.concatenate %a, %b, dim = 1 : (tensor<3x2xf64>, tensor<3x2xf64>) -> tensor<3x4xf64>
  return %c : tensor<3x4xf64>
}

// CHECK-LABEL: func.func @mixed_dims
// CHECK-SAME:    (%[[X:.+]]: tensor<1x2xf64>, %[[Y:.+]]: tensor<1x2xf64>) -> tensor<3x6xf64>
// CHECK-NEXT:    %[[Z:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[RB:.+]] = stablehlo.pad %[[Y]], %[[Z]], low = [0, 1], high = [0, 0], interior = [0, 0] : (tensor<1x2xf64>, tensor<f64>) -> tensor<1x3xf64>
// CHECK-NEXT:    %[[CAT:.+]] = stablehlo.concatenate %[[X]], %[[RB]], dim = 1 : (tensor<1x2xf64>, tensor<1x3xf64>) -> tensor<1x5xf64>
// CHECK-NEXT:    %[[PAD:.+]] = stablehlo.pad %[[CAT]], %[[Z]], low = [1, 1], high = [1, 0], interior = [0, 0] : (tensor<1x5xf64>, tensor<f64>) -> tensor<3x6xf64>
// CHECK-NEXT:    return %[[PAD]] : tensor<3x6xf64>
func.func @mixed_dims(%x: tensor<1x2xf64>, %y: tensor<1x2xf64>) -> tensor<3x6xf64> {
  %z = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %a = stablehlo.pad %x, %z, low = [1, 1], high = [1, 0], interior = [0, 0] : (tensor<1x2xf64>, tensor<f64>) -> tensor<3x3xf64>
  %b = stablehlo.pad %y, %z, low = [1, 1], high = [1, 0], interior = [0, 0] : (tensor<1x2xf64>, tensor<f64>) -> tensor<3x3xf64>
  %c = stablehlo.concatenate %a, %b, dim = 1 : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x6xf64>
  return %c : tensor<3x6xf64>
}

// The rebuilt outer pad has no interior padding, so operands keep theirs.
// CHECK-LABEL: func.func @interior_padding
// CHECK-SAME:    (%[[X:.+]]: tensor<1x2xf64>, %[[Y:.+]]: tensor<1x2xf64>) -> tensor<3x6xf64>
// CHECK-NEXT:    %[[Z:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[RA:.+]] = stablehlo.pad %[[X]], %[[Z]], low = [0, 0], high = [0, 0], interior = [0, 1] : (tensor<1x2xf64>, tensor<f64>) -> tensor<1x3xf64>
// CHECK-NEXT:    %[[RB:.+]] = stablehlo.pad %[[Y]], %[[Z]], low = [0, 0], high = [0, 0], interior = [0, 1] : (tensor<1x2xf64>, tensor<f64>) -> tensor<1x3xf64>
// CHECK-NEXT:    %[[CAT:.+]] = stablehlo.concatenate %[[RA]], %[[RB]], dim = 1 : (tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<1x6xf64>
// CHECK-NEXT:    %[[PAD:.+]] = stablehlo.pad %[[CAT]], %[[Z]], low = [1, 0], high = [1, 0], interior = [0, 0] : (tensor<1x6xf64>, tensor<f64>) -> tensor<3x6xf64>
// CHECK-NEXT:    return %[[PAD]] : tensor<3x6xf64>
func.func @interior_padding(%x: tensor<1x2xf64>, %y: tensor<1x2xf64>) -> tensor<3x6xf64> {
  %z = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %a = stablehlo.pad %x, %z, low = [1, 0], high = [1, 0], interior = [0, 1] : (tensor<1x2xf64>, tensor<f64>) -> tensor<3x3xf64>
  %b = stablehlo.pad %y, %z, low = [1, 0], high = [1, 0], interior = [0, 1] : (tensor<1x2xf64>, tensor<f64>) -> tensor<3x3xf64>
  %c = stablehlo.concatenate %a, %b, dim = 1 : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x6xf64>
  return %c : tensor<3x6xf64>
}
