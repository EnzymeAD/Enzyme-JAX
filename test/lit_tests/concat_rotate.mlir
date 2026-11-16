// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_rotate" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=ROTATE
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_rotate;concat_rotate;concat_slice;concat_fuse;concatenate_op_canon(1024);noop_slice" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=FULL

// Rotates by the same amount along the same dimension commute with a
// concatenate along another dimension.
func.func @merge(%a: tensor<1x4x8xf64>, %b: tensor<1x4x8xf64>, %c: tensor<1x4x8xf64>) -> tensor<3x4x8xf64> {
  %0 = "enzymexla.rotate"(%a) <{amount = 4 : i32, dimension = 2 : i32}> : (tensor<1x4x8xf64>) -> tensor<1x4x8xf64>
  %1 = "enzymexla.rotate"(%b) <{amount = 4 : i32, dimension = 2 : i32}> : (tensor<1x4x8xf64>) -> tensor<1x4x8xf64>
  %2 = "enzymexla.rotate"(%c) <{amount = 4 : i32, dimension = 2 : i32}> : (tensor<1x4x8xf64>) -> tensor<1x4x8xf64>
  %3 = stablehlo.concatenate %0, %1, %2, dim = 0 : (tensor<1x4x8xf64>, tensor<1x4x8xf64>, tensor<1x4x8xf64>) -> tensor<3x4x8xf64>
  return %3 : tensor<3x4x8xf64>
}

// ROTATE:       func.func @merge(%[[ARG0:[a-z0-9_]+]]: tensor<1x4x8xf64>, %[[ARG1:[a-z0-9_]+]]: tensor<1x4x8xf64>, %[[ARG2:[a-z0-9_]+]]: tensor<1x4x8xf64>) -> tensor<3x4x8xf64> {
// ROTATE-NEXT:  %[[V0:[a-z0-9_]+]] = stablehlo.concatenate %[[ARG0]], %[[ARG1]], %[[ARG2]], dim = 0 : (tensor<1x4x8xf64>, tensor<1x4x8xf64>, tensor<1x4x8xf64>) -> tensor<3x4x8xf64>
// ROTATE-NEXT:  %[[V1:[a-z0-9_]+]] = "enzymexla.rotate"(%[[V0]]) <{amount = 4 : i32, dimension = 2 : i32}> : (tensor<3x4x8xf64>) -> tensor<3x4x8xf64>
// ROTATE-NEXT:  return %[[V1]] : tensor<3x4x8xf64>
// ROTATE-NEXT:  }

// Different amounts, a different rotate dimension, or a rotate along the
// concatenation dimension itself are left alone.
func.func @nomerge(%a: tensor<1x4x8xf64>, %b: tensor<1x4x8xf64>, %c: tensor<1x4x8xf64>, %d: tensor<1x4x8xf64>) -> tensor<4x4x8xf64> {
  %0 = "enzymexla.rotate"(%a) <{amount = 4 : i32, dimension = 2 : i32}> : (tensor<1x4x8xf64>) -> tensor<1x4x8xf64>
  %1 = "enzymexla.rotate"(%b) <{amount = 3 : i32, dimension = 2 : i32}> : (tensor<1x4x8xf64>) -> tensor<1x4x8xf64>
  %2 = "enzymexla.rotate"(%c) <{amount = 3 : i32, dimension = 1 : i32}> : (tensor<1x4x8xf64>) -> tensor<1x4x8xf64>
  %3 = "enzymexla.rotate"(%d) <{amount = 1 : i32, dimension = 0 : i32}> : (tensor<1x4x8xf64>) -> tensor<1x4x8xf64>
  %4 = stablehlo.concatenate %0, %1, %2, %3, dim = 0 : (tensor<1x4x8xf64>, tensor<1x4x8xf64>, tensor<1x4x8xf64>, tensor<1x4x8xf64>) -> tensor<4x4x8xf64>
  return %4 : tensor<4x4x8xf64>
}

// ROTATE:       func.func @nomerge(%[[ARG0:[a-z0-9_]+]]: tensor<1x4x8xf64>, %[[ARG1:[a-z0-9_]+]]: tensor<1x4x8xf64>, %[[ARG2:[a-z0-9_]+]]: tensor<1x4x8xf64>, %[[ARG3:[a-z0-9_]+]]: tensor<1x4x8xf64>) -> tensor<4x4x8xf64> {
// ROTATE-NEXT:  %[[V0:[a-z0-9_]+]] = "enzymexla.rotate"(%[[ARG0]]) <{amount = 4 : i32, dimension = 2 : i32}> : (tensor<1x4x8xf64>) -> tensor<1x4x8xf64>
// ROTATE-NEXT:  %[[V1:[a-z0-9_]+]] = "enzymexla.rotate"(%[[ARG1]]) <{amount = 3 : i32, dimension = 2 : i32}> : (tensor<1x4x8xf64>) -> tensor<1x4x8xf64>
// ROTATE-NEXT:  %[[V2:[a-z0-9_]+]] = "enzymexla.rotate"(%[[ARG2]]) <{amount = 3 : i32, dimension = 1 : i32}> : (tensor<1x4x8xf64>) -> tensor<1x4x8xf64>
// ROTATE-NEXT:  %[[V3:[a-z0-9_]+]] = "enzymexla.rotate"(%[[ARG3]]) <{amount = 1 : i32, dimension = 0 : i32}> : (tensor<1x4x8xf64>) -> tensor<1x4x8xf64>
// ROTATE-NEXT:  %[[V4:[a-z0-9_]+]] = stablehlo.concatenate %[[V0]], %[[V1]], %[[V2]], %[[V3]], dim = 0 : (tensor<1x4x8xf64>, tensor<1x4x8xf64>, tensor<1x4x8xf64>, tensor<1x4x8xf64>) -> tensor<4x4x8xf64>
// ROTATE-NEXT:  return %[[V4]] : tensor<4x4x8xf64>
// ROTATE-NEXT:  }

// A circshift (two nested concatenates of slices) computed separately for each
// slice x[k:k+1] and then concatenated back along dim 0 folds back into a
// circshift of x. See EnzymeAD/Reactant.jl#3418.
func.func @perslice(%x: tensor<4x8x8xf64>) -> tensor<4x8x8xf64> {
  %0 = stablehlo.slice %x [0:1, 4:8, 4:8] : (tensor<4x8x8xf64>) -> tensor<1x4x4xf64>
  %1 = stablehlo.slice %x [0:1, 4:8, 0:4] : (tensor<4x8x8xf64>) -> tensor<1x4x4xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 2 : (tensor<1x4x4xf64>, tensor<1x4x4xf64>) -> tensor<1x4x8xf64>
  %3 = stablehlo.slice %x [0:1, 0:4, 4:8] : (tensor<4x8x8xf64>) -> tensor<1x4x4xf64>
  %4 = stablehlo.slice %x [0:1, 0:4, 0:4] : (tensor<4x8x8xf64>) -> tensor<1x4x4xf64>
  %5 = stablehlo.concatenate %3, %4, dim = 2 : (tensor<1x4x4xf64>, tensor<1x4x4xf64>) -> tensor<1x4x8xf64>
  %6 = stablehlo.concatenate %2, %5, dim = 1 : (tensor<1x4x8xf64>, tensor<1x4x8xf64>) -> tensor<1x8x8xf64>
  %7 = stablehlo.slice %x [1:2, 4:8, 4:8] : (tensor<4x8x8xf64>) -> tensor<1x4x4xf64>
  %8 = stablehlo.slice %x [1:2, 4:8, 0:4] : (tensor<4x8x8xf64>) -> tensor<1x4x4xf64>
  %9 = stablehlo.concatenate %7, %8, dim = 2 : (tensor<1x4x4xf64>, tensor<1x4x4xf64>) -> tensor<1x4x8xf64>
  %10 = stablehlo.slice %x [1:2, 0:4, 4:8] : (tensor<4x8x8xf64>) -> tensor<1x4x4xf64>
  %11 = stablehlo.slice %x [1:2, 0:4, 0:4] : (tensor<4x8x8xf64>) -> tensor<1x4x4xf64>
  %12 = stablehlo.concatenate %10, %11, dim = 2 : (tensor<1x4x4xf64>, tensor<1x4x4xf64>) -> tensor<1x4x8xf64>
  %13 = stablehlo.concatenate %9, %12, dim = 1 : (tensor<1x4x8xf64>, tensor<1x4x8xf64>) -> tensor<1x8x8xf64>
  %14 = stablehlo.slice %x [2:3, 4:8, 4:8] : (tensor<4x8x8xf64>) -> tensor<1x4x4xf64>
  %15 = stablehlo.slice %x [2:3, 4:8, 0:4] : (tensor<4x8x8xf64>) -> tensor<1x4x4xf64>
  %16 = stablehlo.concatenate %14, %15, dim = 2 : (tensor<1x4x4xf64>, tensor<1x4x4xf64>) -> tensor<1x4x8xf64>
  %17 = stablehlo.slice %x [2:3, 0:4, 4:8] : (tensor<4x8x8xf64>) -> tensor<1x4x4xf64>
  %18 = stablehlo.slice %x [2:3, 0:4, 0:4] : (tensor<4x8x8xf64>) -> tensor<1x4x4xf64>
  %19 = stablehlo.concatenate %17, %18, dim = 2 : (tensor<1x4x4xf64>, tensor<1x4x4xf64>) -> tensor<1x4x8xf64>
  %20 = stablehlo.concatenate %16, %19, dim = 1 : (tensor<1x4x8xf64>, tensor<1x4x8xf64>) -> tensor<1x8x8xf64>
  %21 = stablehlo.slice %x [3:4, 4:8, 4:8] : (tensor<4x8x8xf64>) -> tensor<1x4x4xf64>
  %22 = stablehlo.slice %x [3:4, 4:8, 0:4] : (tensor<4x8x8xf64>) -> tensor<1x4x4xf64>
  %23 = stablehlo.concatenate %21, %22, dim = 2 : (tensor<1x4x4xf64>, tensor<1x4x4xf64>) -> tensor<1x4x8xf64>
  %24 = stablehlo.slice %x [3:4, 0:4, 4:8] : (tensor<4x8x8xf64>) -> tensor<1x4x4xf64>
  %25 = stablehlo.slice %x [3:4, 0:4, 0:4] : (tensor<4x8x8xf64>) -> tensor<1x4x4xf64>
  %26 = stablehlo.concatenate %24, %25, dim = 2 : (tensor<1x4x4xf64>, tensor<1x4x4xf64>) -> tensor<1x4x8xf64>
  %27 = stablehlo.concatenate %23, %26, dim = 1 : (tensor<1x4x8xf64>, tensor<1x4x8xf64>) -> tensor<1x8x8xf64>
  %28 = stablehlo.concatenate %6, %13, %20, %27, dim = 0 : (tensor<1x8x8xf64>, tensor<1x8x8xf64>, tensor<1x8x8xf64>, tensor<1x8x8xf64>) -> tensor<4x8x8xf64>
  return %28 : tensor<4x8x8xf64>
}

// FULL:       func.func @perslice(%[[ARG0:[a-z0-9_]+]]: tensor<4x8x8xf64>) -> tensor<4x8x8xf64> {
// FULL-NEXT:  %[[V0:[a-z0-9_]+]] = "enzymexla.rotate"(%[[ARG0]]) <{amount = 4 : i32, dimension = 1 : i32}> : (tensor<4x8x8xf64>) -> tensor<4x8x8xf64>
// FULL-NEXT:  %[[V1:[a-z0-9_]+]] = "enzymexla.rotate"(%[[V0]]) <{amount = 4 : i32, dimension = 2 : i32}> : (tensor<4x8x8xf64>) -> tensor<4x8x8xf64>
// FULL-NEXT:  return %[[V1]] : tensor<4x8x8xf64>
// FULL-NEXT:  }
