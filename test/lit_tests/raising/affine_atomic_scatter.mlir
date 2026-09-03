// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// The affine atomic raises as a combining scatter like the others: it is what
// affine-cfg makes of the enzyme atomic before the raiser sees it, and it
// addresses through a map rather than plain indices.

module {
  func.func private @affine_atomic_add(%buf: memref<8xf64, 1>, %idx: memref<16xi32, 1>, %val: memref<16xf64, 1>) {
    affine.parallel (%t) = (0) to (16) {
      %i = affine.load %idx[%t] : memref<16xi32, 1>
      %iidx = arith.index_cast %i : i32 to index
      %v = affine.load %val[%t] : memref<16xf64, 1>
      %old = enzyme.affine_atomic_rmw addf %v, %buf, (affine_map<(d0) -> (d0)>)[%iidx] monotonic : (f64, memref<8xf64, 1>) -> f64
    }
    return
  }
}

// CHECK:  func.func private @affine_atomic_add_raised(%[[a1:.+]]: tensor<8xf64>, %[[a2:.+]]: tensor<16xi32>, %[[a3:.+]]: tensor<16xf64>) -> (tensor<8xf64>, tensor<16xi32>, tensor<16xf64>) {
// CHECK-NEXT:  %[[a4:.+]] = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:  %[[a5:.+]] = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:  %[[a6:.+]] = stablehlo.add %[[a4]], %[[a5]] : tensor<16xi64>
// CHECK-NEXT:  %[[a7:.+]] = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:  %[[a8:.+]] = stablehlo.multiply %[[a6]], %[[a7]] : tensor<16xi64>
// CHECK-NEXT:  %[[a9:.+]] = stablehlo.reshape %[[a2]] : (tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT:  %[[a10:.+]] = stablehlo.convert %[[a9]] : (tensor<16xi32>) -> tensor<16xi64>
// CHECK-NEXT:  %[[a11:.+]] = stablehlo.reshape %[[a3]] : (tensor<16xf64>) -> tensor<16xf64>
// CHECK-NEXT:  %[[a12:.+]] = stablehlo.reshape %[[a10]] : (tensor<16xi64>) -> tensor<16x1xi64>
// CHECK-NEXT:  %[[a13:.+]] = stablehlo.broadcast_in_dim %[[a11]], dims = [0] : (tensor<16xf64>) -> tensor<16xf64>
// CHECK-NEXT:  %[[a14:.+]] = "stablehlo.scatter"(%[[a1]], %[[a12]], %[[a13]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:  ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:  %[[a15:.+]] = stablehlo.add %arg3, %arg4 : tensor<f64>
// CHECK-NEXT:  stablehlo.return %[[a15]] : tensor<f64>
// CHECK-NEXT:  }) : (tensor<8xf64>, tensor<16x1xi64>, tensor<16xf64>) -> tensor<8xf64>
// CHECK-NEXT:  return %[[a14]], %[[a2]], %[[a3]] : tensor<8xf64>, tensor<16xi32>, tensor<16xf64>
// CHECK-NEXT:  }
