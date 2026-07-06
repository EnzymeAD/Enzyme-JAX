// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=binop_pad_to_concat_add" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// CHECK-LABEL: func.func @add_pad_low_high
// CHECK-SAME:    (%[[A:.+]]: tensor<2x3xf32>, %[[Y:.+]]: tensor<2x10xf32>) -> tensor<2x10xf32>
// CHECK-NEXT:    %[[PRE:.+]] = stablehlo.slice %[[Y]] [0:2, 0:4] : (tensor<2x10xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:    %[[MIDSL:.+]] = stablehlo.slice %[[Y]] [0:2, 4:7] : (tensor<2x10xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:    %[[MID:.+]] = stablehlo.add %[[MIDSL]], %[[A]] : tensor<2x3xf32>
// CHECK-NEXT:    %[[POST:.+]] = stablehlo.slice %[[Y]] [0:2, 7:10] : (tensor<2x10xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:    %[[CAT:.+]] = stablehlo.concatenate %[[PRE]], %[[MID]], %[[POST]], dim = 1 : (tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x10xf32>
// CHECK-NEXT:    return %[[CAT]] : tensor<2x10xf32>
func.func @add_pad_low_high(%a: tensor<2x3xf32>, %y: tensor<2x10xf32>) -> tensor<2x10xf32> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %p = stablehlo.pad %a, %zero, low = [0, 4], high = [0, 3], interior = [0, 0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x10xf32>
  %r = stablehlo.add %p, %y : tensor<2x10xf32>
  return %r : tensor<2x10xf32>
}

// CHECK-LABEL: func.func @add_pad_low_only
// CHECK-SAME:    (%[[A:.+]]: tensor<4xf32>, %[[Y:.+]]: tensor<10xf32>) -> tensor<10xf32>
// CHECK-NEXT:    %[[PRE:.+]] = stablehlo.slice %[[Y]] [0:6] : (tensor<10xf32>) -> tensor<6xf32>
// CHECK-NEXT:    %[[MIDSL:.+]] = stablehlo.slice %[[Y]] [6:10] : (tensor<10xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %[[MID:.+]] = stablehlo.add %[[MIDSL]], %[[A]] : tensor<4xf32>
// CHECK-NEXT:    %[[CAT:.+]] = stablehlo.concatenate %[[PRE]], %[[MID]], dim = 0 : (tensor<6xf32>, tensor<4xf32>) -> tensor<10xf32>
// CHECK-NEXT:    return %[[CAT]] : tensor<10xf32>
func.func @add_pad_low_only(%a: tensor<4xf32>, %y: tensor<10xf32>) -> tensor<10xf32> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %p = stablehlo.pad %a, %zero, low = [6], high = [0], interior = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<10xf32>
  %r = stablehlo.add %p, %y : tensor<10xf32>
  return %r : tensor<10xf32>
}

// CHECK-LABEL: func.func @add_pad_interior
// CHECK-SAME:    (%[[A:.+]]: tensor<3xf32>, %[[Y:.+]]: tensor<10xf32>) -> tensor<10xf32>
// CHECK-NEXT:    %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[P:.+]] = stablehlo.pad %[[A]], %[[ZERO]], low = [2], high = [3], interior = [1] : (tensor<3xf32>, tensor<f32>) -> tensor<10xf32>
// CHECK-NEXT:    %[[R:.+]] = stablehlo.add %[[P]], %[[Y]] : tensor<10xf32>
// CHECK-NEXT:    return %[[R]] : tensor<10xf32>
func.func @add_pad_interior(%a: tensor<3xf32>, %y: tensor<10xf32>) -> tensor<10xf32> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %p = stablehlo.pad %a, %zero, low = [2], high = [3], interior = [1] : (tensor<3xf32>, tensor<f32>) -> tensor<10xf32>
  %r = stablehlo.add %p, %y : tensor<10xf32>
  return %r : tensor<10xf32>
}

// CHECK-LABEL: func.func @add_pad_two_dims
// CHECK-SAME:    (%[[A:.+]]: tensor<2x3xf32>, %[[Y:.+]]: tensor<6x10xf32>) -> tensor<6x10xf32>
// CHECK-NEXT:    %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[P:.+]] = stablehlo.pad %[[A]], %[[ZERO]], low = [1, 4], high = [3, 3], interior = [0, 0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<6x10xf32>
// CHECK-NEXT:    %[[R:.+]] = stablehlo.add %[[P]], %[[Y]] : tensor<6x10xf32>
// CHECK-NEXT:    return %[[R]] : tensor<6x10xf32>
func.func @add_pad_two_dims(%a: tensor<2x3xf32>, %y: tensor<6x10xf32>) -> tensor<6x10xf32> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %p = stablehlo.pad %a, %zero, low = [1, 4], high = [3, 3], interior = [0, 0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<6x10xf32>
  %r = stablehlo.add %p, %y : tensor<6x10xf32>
  return %r : tensor<6x10xf32>
}

// Must not fire: the pad's operand is more than half the result.
// CHECK-LABEL: func.func @add_pad_too_large
// CHECK-SAME:    (%[[A:.+]]: tensor<8xf32>, %[[Y:.+]]: tensor<10xf32>) -> tensor<10xf32>
// CHECK-NEXT:    %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[P:.+]] = stablehlo.pad %[[A]], %[[ZERO]], low = [1], high = [1], interior = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<10xf32>
// CHECK-NEXT:    %[[R:.+]] = stablehlo.add %[[P]], %[[Y]] : tensor<10xf32>
// CHECK-NEXT:    return %[[R]] : tensor<10xf32>
func.func @add_pad_too_large(%a: tensor<8xf32>, %y: tensor<10xf32>) -> tensor<10xf32> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %p = stablehlo.pad %a, %zero, low = [1], high = [1], interior = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<10xf32>
  %r = stablehlo.add %p, %y : tensor<10xf32>
  return %r : tensor<10xf32>
}
