// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt="max_constant_expansion=0" --split-input-file | FileCheck %s

module {
  func.func @main() -> tensor<20x180xi64 >{
    %c = stablehlo.constant dense<[0, 20176, 40352, 60528, 80704, 100880, 121056, 141232, 161408, 181584, 201760, 221936, 242112, 262288, 282464, 302640, 322816, 342992, 363168, 383344]> : tensor<20xi64>
    %0 = stablehlo.broadcast_in_dim %c, dims = [0] : (tensor<20xi64>) -> tensor<20x180xi64>
    return %0 : tensor<20x180xi64>
  }
}
// CHECK:       func.func @main() -> tensor<20x180xi64> {
// CHECK-NEXT:    %[[v0:[^ ]*]] = stablehlo.constant dense<20176> : tensor<20x180xi64>
// CHECK-NEXT:    %[[v1:[^ ]*]] = stablehlo.iota dim = 0 : tensor<20x180xi64>
// CHECK-NEXT:    %[[v2:[^ ]*]] = stablehlo.multiply %[[v1]], %[[v0]] : tensor<20x180xi64>
// CHECK-NEXT:    return %[[v2]] : tensor<20x180xi64>
// CHECK-NEXT:  }

// -----

module {
  func.func @main() -> tensor<180x20xi64 >{
    %c_1173 = stablehlo.constant dense<[0, 20176, 40352, 60528, 80704, 100880, 121056, 141232, 161408, 181584, 201760, 221936, 242112, 262288, 282464, 302640, 322816, 342992, 363168, 383344]> : tensor<20xi64>
    %423 = stablehlo.broadcast_in_dim %c_1173, dims = [1] : (tensor<20xi64>) -> tensor<180x20xi64>
    return %423 : tensor<180x20xi64>
  }
}
// CHECK:       func.func @main() -> tensor<180x20xi64> {
// CHECK-NEXT:    %[[v0:[^ ]*]] = stablehlo.constant dense<20176> : tensor<180x20xi64>
// CHECK-NEXT:    %[[v1:[^ ]*]] = stablehlo.iota dim = 1 : tensor<180x20xi64>
// CHECK-NEXT:    %[[v2:[^ ]*]] = stablehlo.multiply %[[v1]], %[[v0]] : tensor<180x20xi64>
// CHECK-NEXT:    return %[[v2]] : tensor<180x20xi64>
// CHECK-NEXT:  }

// -----

module {
  func.func @main() -> tensor<20x180xi64 >{
    %c_1173 = stablehlo.constant dense<[1, 20177, 40353, 60529, 80705, 100881, 121057, 141233, 161409, 181585, 201761, 221937, 242113, 262289, 282465, 302641, 322817, 342993, 363169, 383345]> : tensor<20xi64>
    %423 = stablehlo.broadcast_in_dim %c_1173, dims = [0] : (tensor<20xi64>) -> tensor<20x180xi64>
    return %423 : tensor<20x180xi64>
  }
}
// CHECK:       func.func @main() -> tensor<20x180xi64> {
// CHECK-NEXT:    %[[v0:[^ ]*]] = stablehlo.constant dense<1> : tensor<20x180xi64>
// CHECK-NEXT:    %[[v1:[^ ]*]] = stablehlo.constant dense<20176> : tensor<20x180xi64>
// CHECK-NEXT:    %[[v2:[^ ]*]] = stablehlo.iota dim = 0 : tensor<20x180xi64>
// CHECK-NEXT:    %[[v3:[^ ]*]] = stablehlo.multiply %[[v2]], %[[v1]] : tensor<20x180xi64>
// CHECK-NEXT:    %[[v4:[^ ]*]] = stablehlo.add %[[v0]], %[[v3]] : tensor<20x180xi64>
// CHECK-NEXT:    return %[[v4]] : tensor<20x180xi64>
// CHECK-NEXT:  }

// -----

func.func @arith_broadcast_dim0() -> tensor<4x8xi32> {
  %c = stablehlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %0 = stablehlo.broadcast_in_dim %c, dims = [0] : (tensor<4xi32>) -> tensor<4x8xi32>
  return %0 : tensor<4x8xi32>
}

// CHECK-LABEL: @arith_broadcast_dim0
// CHECK: stablehlo.iota dim = 0 : tensor<4x8xi32>

// -----

func.func @arith_broadcast_dim1() -> tensor<8x4xi32> {
  %c = stablehlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %0 = stablehlo.broadcast_in_dim %c, dims = [1] : (tensor<4xi32>) -> tensor<8x4xi32>
  return %0 : tensor<8x4xi32>
}

// CHECK-LABEL: @arith_broadcast_dim1
// CHECK: stablehlo.iota dim = 1 : tensor<8x4xi32>

// -----

func.func @arith_broadcast_dim1_3d() -> tensor<5x4x6xi32> {
  %c = stablehlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %0 = stablehlo.broadcast_in_dim %c, dims = [1] : (tensor<4xi32>) -> tensor<5x4x6xi32>
  return %0 : tensor<5x4x6xi32>
}

// CHECK-LABEL: @arith_broadcast_dim1_3d
// CHECK: stablehlo.iota dim = 1 : tensor<5x4x6xi32>

// -----

func.func @arith_broadcast_nonzero_start() -> tensor<4x8xi32> {
  %c = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %0 = stablehlo.broadcast_in_dim %c, dims = [0] : (tensor<4xi32>) -> tensor<4x8xi32>
  return %0 : tensor<4x8xi32>
}

// CHECK-LABEL: @arith_broadcast_nonzero_start
// CHECK: stablehlo.iota dim = 0 : tensor<4x8xi32>
