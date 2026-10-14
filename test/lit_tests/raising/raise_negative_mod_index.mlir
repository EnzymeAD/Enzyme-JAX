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

// The index computation folds to one constant: for %i = 3 the remainders
// -2, -1 and 0 become 1, 2 and 0 -- only the negative ones get the modulus
// added, and the exact multiple stays 0.
// CHECK-LABEL: func.func private @negmod_raised(%arg0: tensor<4x3xf64>, %arg1: tensor<4x3xf64>) -> (tensor<4x3xf64>, tensor<4x3xf64>) {
// CHECK-NEXT: %c = stablehlo.constant dense<{{\[\[\[}}0, 0], [0, 1], [0, 2]], {{\[\[}}1, 0], [1, 1], [1, 2]], {{\[\[}}2, 0], [2, 1], [2, 2]], {{\[\[}}3, 1], [3, 2], [3, 0]]]> : tensor<4x3x2xi64>
// CHECK-NEXT: %0 = "stablehlo.scatter"(%arg0, %c, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = true}> ({
// CHECK-NEXT: ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT: stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT: }) : (tensor<4x3xf64>, tensor<4x3x2xi64>, tensor<4x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: return %0, %arg1 : tensor<4x3xf64>, tensor<4x3xf64>
// CHECK-NEXT: }
