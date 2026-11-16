// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=cse_dot_general_transposed;transpose_transpose" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Both contract the last dimension of %x with that of %b, reading %x through
// different transposes: the second is a transpose of the first.
func.func @layouts(%x: tensor<5x7x6xf64>, %b: tensor<4x6xf64>) -> (tensor<7x5x4xf64>, tensor<7x4x5xf64>) {
  %t0 = stablehlo.transpose %x, dims = [2, 0, 1] : (tensor<5x7x6xf64>) -> tensor<6x5x7xf64>
  %d0 = stablehlo.dot_general %t0, %b, contracting_dims = [0] x [1] : (tensor<6x5x7xf64>, tensor<4x6xf64>) -> tensor<5x7x4xf64>
  %r0 = stablehlo.transpose %d0, dims = [1, 0, 2] : (tensor<5x7x4xf64>) -> tensor<7x5x4xf64>
  %t1 = stablehlo.transpose %x, dims = [1, 2, 0] : (tensor<5x7x6xf64>) -> tensor<7x6x5xf64>
  %d1 = stablehlo.dot_general %t1, %b, contracting_dims = [1] x [1] : (tensor<7x6x5xf64>, tensor<4x6xf64>) -> tensor<7x5x4xf64>
  %r1 = stablehlo.transpose %d1, dims = [0, 2, 1] : (tensor<7x5x4xf64>) -> tensor<7x4x5xf64>
  return %r0, %r1 : tensor<7x5x4xf64>, tensor<7x4x5xf64>
}

// CHECK-LABEL: func.func @layouts
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<5x7x6xf64>) -> tensor<6x5x7xf64>
// CHECK-NEXT:    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [0] x [1] : (tensor<6x5x7xf64>, tensor<4x6xf64>) -> tensor<5x7x4xf64>
// CHECK-NEXT:    %2 = stablehlo.transpose %1, dims = [1, 0, 2] : (tensor<5x7x4xf64>) -> tensor<7x5x4xf64>
// CHECK-NEXT:    %3 = stablehlo.transpose %1, dims = [1, 2, 0] : (tensor<5x7x4xf64>) -> tensor<7x4x5xf64>
// CHECK-NEXT:    return %2, %3 : tensor<7x5x4xf64>, tensor<7x4x5xf64>
// CHECK-NEXT:  }

// The operands in the other order, with a batch dimension.
func.func @swapped(%x: tensor<3x5x6xf64>, %b: tensor<3x4x6xf64>) -> (tensor<3x5x4xf64>, tensor<3x4x5xf64>) {
  %d0 = stablehlo.dot_general %x, %b, batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<3x5x6xf64>, tensor<3x4x6xf64>) -> tensor<3x5x4xf64>
  %d1 = stablehlo.dot_general %b, %x, batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<3x4x6xf64>, tensor<3x5x6xf64>) -> tensor<3x4x5xf64>
  return %d0, %d1 : tensor<3x5x4xf64>, tensor<3x4x5xf64>
}

// CHECK-LABEL: func.func @swapped
// CHECK-NEXT:    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<3x5x6xf64>, tensor<3x4x6xf64>) -> tensor<3x5x4xf64>
// CHECK-NEXT:    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<3x5x4xf64>) -> tensor<3x4x5xf64>
// CHECK-NEXT:    return %0, %1 : tensor<3x5x4xf64>, tensor<3x4x5xf64>
// CHECK-NEXT:  }

// Different dimensions of %x are contracted: not the same value.
func.func @different(%x: tensor<6x6xf64>, %b: tensor<4x6xf64>) -> (tensor<6x4xf64>, tensor<6x4xf64>) {
  %d0 = stablehlo.dot_general %x, %b, contracting_dims = [1] x [1] : (tensor<6x6xf64>, tensor<4x6xf64>) -> tensor<6x4xf64>
  %t = stablehlo.transpose %x, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
  %d1 = stablehlo.dot_general %t, %b, contracting_dims = [1] x [1] : (tensor<6x6xf64>, tensor<4x6xf64>) -> tensor<6x4xf64>
  return %d0, %d1 : tensor<6x4xf64>, tensor<6x4xf64>
}

// CHECK-LABEL: func.func @different
// CHECK-NEXT:    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1] : (tensor<6x6xf64>, tensor<4x6xf64>) -> tensor<6x4xf64>
// CHECK-NEXT:    %1 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
// CHECK-NEXT:    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1] : (tensor<6x6xf64>, tensor<4x6xf64>) -> tensor<6x4xf64>
// CHECK-NEXT:    return %0, %2 : tensor<6x4xf64>, tensor<6x4xf64>
// CHECK-NEXT:  }
