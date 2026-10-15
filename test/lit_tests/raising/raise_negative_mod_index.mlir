// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --enzyme-hlo-opt | FileCheck %s

// An affine mod of a negative exact multiple is 0, not the modulus:
// (%j + (%i * -8) floordiv 3) mod 3 is %j for %i < 3 (the floordiv gives
// -3 and -6 there) and (%j + 1) mod 3 for %i = 3. The lowering used to add
// the modulus whenever the dividend was negative, so the (1, 0) and (2, 0)
// lanes indexed slot 3 of a 3-slot row instead of slot 0: out of the buffer,
// and the scatter dropped their stores.
func.func @negmod(%out: memref<4x3xf64, 1>, %in: memref<4x3xf64, 1>) {
  affine.parallel (%i, %j) = (0, 0) to (4, 3) {
    %v = affine.load %in[%i, %j] : memref<4x3xf64, 1>
    affine.store %v, %out[%i, (%j + (%i * -8) floordiv 3) mod 3] : memref<4x3xf64, 1>
  }
  return
}

// The remainders fold to constants: in column 3 (%i = 3) they are -2, -1 and
// 0, and only the negative ones get the modulus added -- the exact multiple
// stays 0.
// CHECK-LABEL: func.func private @negmod_raised(%arg0: tensor<4x3xf64>, %arg1: tensor<4x3xf64>) -> (tensor<4x3xf64>, tensor<4x3xf64>) {
// CHECK-NEXT: %c = stablehlo.constant dense<{{\[\[}}[0], [0], [0]], {{\[\[}}1], [1], [1]], {{\[\[}}2], [2], [2]], {{\[\[}}3], [3], [3]]]> : tensor<4x3x1xi64>
// CHECK-NEXT: %c_0 = stablehlo.constant dense<3> : tensor<3x4xi64>
// CHECK-NEXT: %c_1 = stablehlo.constant dense<{{\[\[}}0, 0, 0, -2], [1, -2, -2, -1], [2, -1, -1, 0]]> : tensor<3x4xi64>
// CHECK-NEXT: %c_2 = stablehlo.constant dense<{{\[\[}}false, false, false, true], [false, true, true, true], [false, true, true, false]]> : tensor<3x4xi1>
// CHECK-NEXT: %0 = stablehlo.add %c_1, %c_0 : tensor<3x4xi64>
// CHECK-NEXT: %1 = stablehlo.select %c_2, %0, %c_1 : tensor<3x4xi1>, tensor<3x4xi64>
// CHECK-NEXT: %2 = stablehlo.broadcast_in_dim %1, dims = [1, 0] : (tensor<3x4xi64>) -> tensor<4x3x1xi64>
// CHECK-NEXT: %3 = stablehlo.concatenate %c, %2, dim = 2 : (tensor<4x3x1xi64>, tensor<4x3x1xi64>) -> tensor<4x3x2xi64>
// CHECK-NEXT: %4 = "stablehlo.scatter"(%arg0, %3, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = true}> ({
// CHECK-NEXT: ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT: stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT: }) : (tensor<4x3xf64>, tensor<4x3x2xi64>, tensor<4x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: return %4, %arg1 : tensor<4x3xf64>, tensor<4x3xf64>
// CHECK-NEXT: }
