// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=dot_general_outer_product_reassociate" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

func.func @rhs_first(%a: tensor<3x4xf64>, %u: tensor<4xf64>, %v: tensor<5xf64>) -> tensor<3x5xf64> {
  %o = stablehlo.dot_general %u, %v, contracting_dims = [] x [] : (tensor<4xf64>, tensor<5xf64>) -> tensor<4x5xf64>
  %r = stablehlo.dot_general %a, %o, contracting_dims = [1] x [0] : (tensor<3x4xf64>, tensor<4x5xf64>) -> tensor<3x5xf64>
  return %r : tensor<3x5xf64>
}

func.func @rhs_second(%a: tensor<3x5xf64>, %u: tensor<4xf64>, %v: tensor<5xf64>) -> tensor<3x4xf64> {
  %o = stablehlo.dot_general %u, %v, contracting_dims = [] x [] : (tensor<4xf64>, tensor<5xf64>) -> tensor<4x5xf64>
  %r = stablehlo.dot_general %a, %o, contracting_dims = [1] x [1] : (tensor<3x5xf64>, tensor<4x5xf64>) -> tensor<3x4xf64>
  return %r : tensor<3x4xf64>
}

func.func @lhs(%u: tensor<4xf64>, %v: tensor<5xf64>, %b: tensor<5x6xf64>) -> (tensor<4x6xf64>, tensor<4x5xf64>) {
  %o = stablehlo.dot_general %u, %v, contracting_dims = [] x [] : (tensor<4xf64>, tensor<5xf64>) -> tensor<4x5xf64>
  %r = stablehlo.dot_general %o, %b, contracting_dims = [1] x [0] : (tensor<4x5xf64>, tensor<5x6xf64>) -> tensor<4x6xf64>
  return %r, %o : tensor<4x6xf64>, tensor<4x5xf64>
}

func.func @not_outer(%a: tensor<3x4xf64>, %x: tensor<4x2xf64>, %y: tensor<2x5xf64>) -> tensor<3x5xf64> {
  %o = stablehlo.dot_general %x, %y, contracting_dims = [1] x [0] : (tensor<4x2xf64>, tensor<2x5xf64>) -> tensor<4x5xf64>
  %r = stablehlo.dot_general %a, %o, contracting_dims = [1] x [0] : (tensor<3x4xf64>, tensor<4x5xf64>) -> tensor<3x5xf64>
  return %r : tensor<3x5xf64>
}

func.func @batched(%a: tensor<2x3x4xf64>, %u: tensor<4xf64>, %v: tensor<2xf64>) -> tensor<2x3xf64> {
  %o = stablehlo.dot_general %u, %v, contracting_dims = [] x [] : (tensor<4xf64>, tensor<2xf64>) -> tensor<4x2xf64>
  %r = stablehlo.dot_general %a, %o, batching_dims = [0] x [1], contracting_dims = [2] x [0] : (tensor<2x3x4xf64>, tensor<4x2xf64>) -> tensor<2x3xf64>
  return %r : tensor<2x3xf64>
}

// CHECK-LABEL: func.func @rhs_first(
// CHECK-SAME:      %[[A:.+]]: tensor<3x4xf64>, %[[U:.+]]: tensor<4xf64>, %[[V:.+]]: tensor<5xf64>)
// CHECK-NEXT:    %[[AU:.+]] = stablehlo.dot_general %[[A]], %[[U]], contracting_dims = [1] x [0] : (tensor<3x4xf64>, tensor<4xf64>) -> tensor<3xf64>
// CHECK-NEXT:    %[[R:.+]] = stablehlo.dot_general %[[AU]], %[[V]], contracting_dims = [] x [] : (tensor<3xf64>, tensor<5xf64>) -> tensor<3x5xf64>
// CHECK-NEXT:    return %[[R]] : tensor<3x5xf64>

// CHECK-LABEL: func.func @rhs_second(
// CHECK-SAME:      %[[A:.+]]: tensor<3x5xf64>, %[[U:.+]]: tensor<4xf64>, %[[V:.+]]: tensor<5xf64>)
// CHECK-NEXT:    %[[AV:.+]] = stablehlo.dot_general %[[A]], %[[V]], contracting_dims = [1] x [0] : (tensor<3x5xf64>, tensor<5xf64>) -> tensor<3xf64>
// CHECK-NEXT:    %[[R:.+]] = stablehlo.dot_general %[[AV]], %[[U]], contracting_dims = [] x [] : (tensor<3xf64>, tensor<4xf64>) -> tensor<3x4xf64>
// CHECK-NEXT:    return %[[R]] : tensor<3x4xf64>

// CHECK-LABEL: func.func @lhs(
// CHECK-SAME:      %[[U:.+]]: tensor<4xf64>, %[[V:.+]]: tensor<5xf64>, %[[B:.+]]: tensor<5x6xf64>)
// CHECK-NEXT:    %[[O:.+]] = stablehlo.dot_general %[[U]], %[[V]], contracting_dims = [] x [] : (tensor<4xf64>, tensor<5xf64>) -> tensor<4x5xf64>
// CHECK-NEXT:    %[[VB:.+]] = stablehlo.dot_general %[[V]], %[[B]], contracting_dims = [0] x [0] : (tensor<5xf64>, tensor<5x6xf64>) -> tensor<6xf64>
// CHECK-NEXT:    %[[R:.+]] = stablehlo.dot_general %[[U]], %[[VB]], contracting_dims = [] x [] : (tensor<4xf64>, tensor<6xf64>) -> tensor<4x6xf64>
// CHECK-NEXT:    return %[[R]], %[[O]] : tensor<4x6xf64>, tensor<4x5xf64>

// CHECK-LABEL: func.func @not_outer(
// CHECK-SAME:      %[[A:.+]]: tensor<3x4xf64>, %[[X:.+]]: tensor<4x2xf64>, %[[Y:.+]]: tensor<2x5xf64>)
// CHECK-NEXT:    %[[O:.+]] = stablehlo.dot_general %[[X]], %[[Y]], contracting_dims = [1] x [0] : (tensor<4x2xf64>, tensor<2x5xf64>) -> tensor<4x5xf64>
// CHECK-NEXT:    %[[R:.+]] = stablehlo.dot_general %[[A]], %[[O]], contracting_dims = [1] x [0] : (tensor<3x4xf64>, tensor<4x5xf64>) -> tensor<3x5xf64>
// CHECK-NEXT:    return %[[R]] : tensor<3x5xf64>

// CHECK-LABEL: func.func @batched(
// CHECK-SAME:      %[[A:.+]]: tensor<2x3x4xf64>, %[[U:.+]]: tensor<4xf64>, %[[V:.+]]: tensor<2xf64>)
// CHECK-NEXT:    %[[O:.+]] = stablehlo.dot_general %[[U]], %[[V]], contracting_dims = [] x [] : (tensor<4xf64>, tensor<2xf64>) -> tensor<4x2xf64>
// CHECK-NEXT:    %[[R:.+]] = stablehlo.dot_general %[[A]], %[[O]], batching_dims = [0] x [1], contracting_dims = [2] x [0] : (tensor<2x3x4xf64>, tensor<4x2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    return %[[R]] : tensor<2x3xf64>
