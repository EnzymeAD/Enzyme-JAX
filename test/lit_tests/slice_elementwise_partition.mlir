// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=slice_elementwise_partition;slice_concat" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// Products of stacked columns read back one column at a time: each column is
// computed from its own pieces, and the stacked operands are never formed.
func.func @columns(%a: tensor<8x1xf64>, %b: tensor<8x1xf64>, %c: tensor<8x1xf64>, %d: tensor<8x1xf64>, %e: tensor<8x2xf64>) -> (tensor<8x1xf64>, tensor<8x1xf64>, tensor<8x2xf64>) {
  %x = stablehlo.concatenate %a, %b, %e, dim = 1 : (tensor<8x1xf64>, tensor<8x1xf64>, tensor<8x2xf64>) -> tensor<8x4xf64>
  %y = stablehlo.concatenate %c, %d, %e, dim = 1 : (tensor<8x1xf64>, tensor<8x1xf64>, tensor<8x2xf64>) -> tensor<8x4xf64>
  %m = stablehlo.multiply %x, %y : tensor<8x4xf64>
  %s = stablehlo.add %m, %x : tensor<8x4xf64>
  %0 = stablehlo.slice %s [0:8, 0:1] : (tensor<8x4xf64>) -> tensor<8x1xf64>
  %1 = stablehlo.slice %s [0:8, 1:2] : (tensor<8x4xf64>) -> tensor<8x1xf64>
  %2 = stablehlo.slice %s [0:8, 2:4] : (tensor<8x4xf64>) -> tensor<8x2xf64>
  return %0, %1, %2 : tensor<8x1xf64>, tensor<8x1xf64>, tensor<8x2xf64>
}

// CHECK:  func.func @columns(%arg0: tensor<8x1xf64>, %arg1: tensor<8x1xf64>, %arg2: tensor<8x1xf64>, %arg3: tensor<8x1xf64>, %arg4: tensor<8x2xf64>) -> (tensor<8x1xf64>, tensor<8x1xf64>, tensor<8x2xf64>) {
// CHECK-NEXT:    %0 = stablehlo.multiply %arg0, %arg2 : tensor<8x1xf64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %arg0 : tensor<8x1xf64>
// CHECK-NEXT:    %2 = stablehlo.multiply %arg1, %arg3 : tensor<8x1xf64>
// CHECK-NEXT:    %3 = stablehlo.add %2, %arg1 : tensor<8x1xf64>
// CHECK-NEXT:    %4 = stablehlo.multiply %arg4, %arg4 : tensor<8x2xf64>
// CHECK-NEXT:    %5 = stablehlo.add %4, %arg4 : tensor<8x2xf64>
// CHECK-NEXT:    return %1, %3, %5 : tensor<8x1xf64>, tensor<8x1xf64>, tensor<8x2xf64>
// CHECK-NEXT:  }

// -----

// Overlapping slices would compute some columns twice; a slice across two
// pieces of a concatenate does not fold; a value also read whole is formed
// anyway.
func.func @kept(%a: tensor<8x1xf64>, %b: tensor<8x3xf64>) -> (tensor<8x2xf64>, tensor<8x2xf64>, tensor<8x2xf64>, tensor<8x2xf64>, tensor<8x4xf64>, tensor<8x1xf64>) {
  %x = stablehlo.concatenate %a, %b, dim = 1 : (tensor<8x1xf64>, tensor<8x3xf64>) -> tensor<8x4xf64>
  %n = stablehlo.negate %x : tensor<8x4xf64>
  %0 = stablehlo.slice %n [0:8, 1:3] : (tensor<8x4xf64>) -> tensor<8x2xf64>
  %1 = stablehlo.slice %n [0:8, 2:4] : (tensor<8x4xf64>) -> tensor<8x2xf64>
  %m = stablehlo.multiply %x, %x : tensor<8x4xf64>
  %2 = stablehlo.slice %m [0:8, 0:2] : (tensor<8x4xf64>) -> tensor<8x2xf64>
  %3 = stablehlo.slice %m [0:8, 2:4] : (tensor<8x4xf64>) -> tensor<8x2xf64>
  %s = stablehlo.sine %x : tensor<8x4xf64>
  %4 = stablehlo.slice %s [0:8, 0:1] : (tensor<8x4xf64>) -> tensor<8x1xf64>
  return %0, %1, %2, %3, %s, %4 : tensor<8x2xf64>, tensor<8x2xf64>, tensor<8x2xf64>, tensor<8x2xf64>, tensor<8x4xf64>, tensor<8x1xf64>
}

// CHECK:  func.func @kept(%arg0: tensor<8x1xf64>, %arg1: tensor<8x3xf64>) -> (tensor<8x2xf64>, tensor<8x2xf64>, tensor<8x2xf64>, tensor<8x2xf64>, tensor<8x4xf64>, tensor<8x1xf64>) {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<8x1xf64>, tensor<8x3xf64>) -> tensor<8x4xf64>
// CHECK-NEXT:    %1 = stablehlo.negate %0 : tensor<8x4xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %1 [0:8, 1:3] : (tensor<8x4xf64>) -> tensor<8x2xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %1 [0:8, 2:4] : (tensor<8x4xf64>) -> tensor<8x2xf64>
// CHECK-NEXT:    %4 = stablehlo.multiply %0, %0 : tensor<8x4xf64>
// CHECK-NEXT:    %5 = stablehlo.slice %4 [0:8, 0:2] : (tensor<8x4xf64>) -> tensor<8x2xf64>
// CHECK-NEXT:    %6 = stablehlo.slice %4 [0:8, 2:4] : (tensor<8x4xf64>) -> tensor<8x2xf64>
// CHECK-NEXT:    %7 = stablehlo.sine %0 : tensor<8x4xf64>
// CHECK-NEXT:    %8 = stablehlo.slice %7 [0:8, 0:1] : (tensor<8x4xf64>) -> tensor<8x1xf64>
// CHECK-NEXT:    return %2, %3, %5, %6, %7, %8 : tensor<8x2xf64>, tensor<8x2xf64>, tensor<8x2xf64>, tensor<8x2xf64>, tensor<8x4xf64>, tensor<8x1xf64>
// CHECK-NEXT:  }

// -----

// An outer product of broadcasts read back one row at a time: nothing is
// stacked, so it is kept whole.
func.func @broadcasts(%a: tensor<5xf64>, %b: tensor<5xf64>) -> (tensor<1x5xf64>, tensor<4x5xf64>) {
  %x = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<5xf64>) -> tensor<5x5xf64>
  %y = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<5xf64>) -> tensor<5x5xf64>
  %m = stablehlo.multiply %x, %y : tensor<5x5xf64>
  %0 = stablehlo.slice %m [0:1, 0:5] : (tensor<5x5xf64>) -> tensor<1x5xf64>
  %1 = stablehlo.slice %m [1:5, 0:5] : (tensor<5x5xf64>) -> tensor<4x5xf64>
  return %0, %1 : tensor<1x5xf64>, tensor<4x5xf64>
}

// CHECK:  func.func @broadcasts(%arg0: tensor<5xf64>, %arg1: tensor<5xf64>) -> (tensor<1x5xf64>, tensor<4x5xf64>) {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<5xf64>) -> tensor<5x5xf64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<5xf64>) -> tensor<5x5xf64>
// CHECK-NEXT:    %2 = stablehlo.multiply %0, %1 : tensor<5x5xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %2 [0:1, 0:5] : (tensor<5x5xf64>) -> tensor<1x5xf64>
// CHECK-NEXT:    %4 = stablehlo.slice %2 [1:5, 0:5] : (tensor<5x5xf64>) -> tensor<4x5xf64>
// CHECK-NEXT:    return %3, %4 : tensor<1x5xf64>, tensor<4x5xf64>
// CHECK-NEXT:  }
