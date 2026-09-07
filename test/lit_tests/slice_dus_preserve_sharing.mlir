// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Both updates write columns [0,4), while the read uses column 7. Pulling the
// read back through the final update would add a second user to %first, keeping
// an old buffer version live across the next write. Preserve the single-use
// chain so the backend can perform the writes in place.
// CHECK-LABEL: func.func @keep_dus_chain_single_use
// CHECK: %[[FIRST:.*]] = stablehlo.dynamic_update_slice
// CHECK: %[[SECOND:.*]] = stablehlo.dynamic_update_slice %[[FIRST]],
// CHECK: %[[READ:.*]] = stablehlo.slice %[[SECOND]] [0:2, 7:8]
// CHECK: return %[[SECOND]], %[[READ]]
func.func @keep_dus_chain_single_use(%dest: tensor<16x8xi32>, %update: tensor<2x4xi32>) -> (tensor<16x8xi32>, tensor<2x1xi32>) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %first = stablehlo.dynamic_update_slice %dest, %update, %c0, %c0 : (tensor<16x8xi32>, tensor<2x4xi32>, tensor<i32>, tensor<i32>) -> tensor<16x8xi32>
  %second = stablehlo.dynamic_update_slice %first, %update, %c4, %c0 : (tensor<16x8xi32>, tensor<2x4xi32>, tensor<i32>, tensor<i32>) -> tensor<16x8xi32>
  %read = stablehlo.slice %second [0:2, 7:8] : (tensor<16x8xi32>) -> tensor<2x1xi32>
  return %second, %read : tensor<16x8xi32>, tensor<2x1xi32>
}
