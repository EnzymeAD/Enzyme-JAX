// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=rotate_rotate" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Two rotations along the same dimension compose into one.
func.func @same_dim(%x: tensor<4x8xf64>) -> tensor<4x8xf64> {
  %0 = "enzymexla.rotate"(%x) <{amount = 3 : i32, dimension = 1 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
  %1 = "enzymexla.rotate"(%0) <{amount = 2 : i32, dimension = 1 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
  return %1 : tensor<4x8xf64>
}

// CHECK:       func.func @same_dim(%[[ARG0:[a-z0-9_]+]]: tensor<4x8xf64>) -> tensor<4x8xf64> {
// CHECK-NEXT:  %[[V0:[a-z0-9_]+]] = "enzymexla.rotate"(%[[ARG0]]) <{amount = 5 : i32, dimension = 1 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:  return %[[V0]] : tensor<4x8xf64>
// CHECK-NEXT:  }

// The combined amount wraps around the dimension size.
func.func @wrap_around(%x: tensor<4x8xf64>) -> tensor<4x8xf64> {
  %0 = "enzymexla.rotate"(%x) <{amount = 6 : i32, dimension = 1 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
  %1 = "enzymexla.rotate"(%0) <{amount = 5 : i32, dimension = 1 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
  return %1 : tensor<4x8xf64>
}

// CHECK:       func.func @wrap_around(%[[ARG0:[a-z0-9_]+]]: tensor<4x8xf64>) -> tensor<4x8xf64> {
// CHECK-NEXT:  %[[V0:[a-z0-9_]+]] = "enzymexla.rotate"(%[[ARG0]]) <{amount = 3 : i32, dimension = 1 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:  return %[[V0]] : tensor<4x8xf64>
// CHECK-NEXT:  }

// A full turn is the identity.
func.func @full_turn(%x: tensor<4x8xf64>) -> tensor<4x8xf64> {
  %0 = "enzymexla.rotate"(%x) <{amount = 3 : i32, dimension = 1 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
  %1 = "enzymexla.rotate"(%0) <{amount = 5 : i32, dimension = 1 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
  return %1 : tensor<4x8xf64>
}

// CHECK:       func.func @full_turn(%[[ARG0:[a-z0-9_]+]]: tensor<4x8xf64>) -> tensor<4x8xf64> {
// CHECK-NEXT:  return %[[ARG0]] : tensor<4x8xf64>
// CHECK-NEXT:  }

// Rotations along different dimensions are left alone.
func.func @different_dims(%x: tensor<4x8xf64>) -> tensor<4x8xf64> {
  %0 = "enzymexla.rotate"(%x) <{amount = 3 : i32, dimension = 1 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
  %1 = "enzymexla.rotate"(%0) <{amount = 2 : i32, dimension = 0 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
  return %1 : tensor<4x8xf64>
}

// CHECK:       func.func @different_dims(%[[ARG0:[a-z0-9_]+]]: tensor<4x8xf64>) -> tensor<4x8xf64> {
// CHECK-NEXT:  %[[V0:[a-z0-9_]+]] = "enzymexla.rotate"(%[[ARG0]]) <{amount = 3 : i32, dimension = 1 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:  %[[V1:[a-z0-9_]+]] = "enzymexla.rotate"(%[[V0]]) <{amount = 2 : i32, dimension = 0 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:  return %[[V1]] : tensor<4x8xf64>
// CHECK-NEXT:  }

// The inner rotate may keep other users; only the outer one is rewritten.
func.func @multi_use(%x: tensor<4x8xf64>) -> (tensor<4x8xf64>, tensor<4x8xf64>) {
  %0 = "enzymexla.rotate"(%x) <{amount = 3 : i32, dimension = 1 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
  %1 = "enzymexla.rotate"(%0) <{amount = 2 : i32, dimension = 1 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
  return %0, %1 : tensor<4x8xf64>, tensor<4x8xf64>
}

// CHECK:       func.func @multi_use(%[[ARG0:[a-z0-9_]+]]: tensor<4x8xf64>) -> (tensor<4x8xf64>, tensor<4x8xf64>) {
// CHECK-NEXT:  %[[V0:[a-z0-9_]+]] = "enzymexla.rotate"(%[[ARG0]]) <{amount = 3 : i32, dimension = 1 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:  %[[V1:[a-z0-9_]+]] = "enzymexla.rotate"(%[[ARG0]]) <{amount = 5 : i32, dimension = 1 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:  return %[[V0]], %[[V1]] : tensor<4x8xf64>, tensor<4x8xf64>
// CHECK-NEXT:  }
