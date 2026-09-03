// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// The enzyme atomic raises like the memref one: an accumulation whose old
// value is unused becomes a combining scatter. It is the form an llvm
// atomicrmw takes when the ordering it carried has to survive the rewrite.

module {
  func.func private @atomic_add(%buf: memref<8xf64, 1>, %idx: memref<16xi32, 1>, %val: memref<16xf64, 1>) {
    affine.parallel (%t) = (0) to (16) {
      %i = affine.load %idx[%t] : memref<16xi32, 1>
      %iidx = arith.index_cast %i : i32 to index
      %v = affine.load %val[%t] : memref<16xf64, 1>
      %old = enzyme.atomic_rmw addf %v, %buf[%iidx] monotonic : (f64, memref<8xf64, 1>) -> f64
    }
    return
  }
}

// CHECK:  func.func private @atomic_add_raised(%[[r1:.+]]: tensor<8xf64>, %[[r2:.+]]: tensor<16xi32>, %[[r3:.+]]: tensor<16xf64>) -> (tensor<8xf64>, tensor<16xi32>, tensor<16xf64>) {
// CHECK-NEXT:  %[[r4:.+]] = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:  %[[r5:.+]] = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:  %[[r6:.+]] = stablehlo.add %[[r4]], %[[r5]] : tensor<16xi64>
// CHECK-NEXT:  %[[r7:.+]] = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:  %[[r8:.+]] = stablehlo.multiply %[[r6]], %[[r7]] : tensor<16xi64>
// CHECK-NEXT:  %[[r9:.+]] = stablehlo.reshape %[[r2]] : (tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT:  %[[r10:.+]] = stablehlo.convert %[[r9]] : (tensor<16xi32>) -> tensor<16xi64>
// CHECK-NEXT:  %[[r11:.+]] = stablehlo.reshape %[[r3]] : (tensor<16xf64>) -> tensor<16xf64>
// CHECK-NEXT:  %[[r12:.+]] = stablehlo.reshape %[[r10]] : (tensor<16xi64>) -> tensor<16x1xi64>
// CHECK-NEXT:  %[[r13:.+]] = stablehlo.broadcast_in_dim %[[r11]], dims = [0] : (tensor<16xf64>) -> tensor<16xf64>
// CHECK-NEXT:  %[[r14:.+]] = "stablehlo.scatter"(%[[r1]], %[[r12]], %[[r13]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:  ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:  %[[r15:.+]] = stablehlo.add %arg3, %arg4 : tensor<f64>
// CHECK-NEXT:  stablehlo.return %[[r15]] : tensor<f64>
// CHECK-NEXT:  }) : (tensor<8xf64>, tensor<16x1xi64>, tensor<16xf64>) -> tensor<8xf64>
// CHECK-NEXT:  return %[[r14]], %[[r2]], %[[r3]] : tensor<8xf64>, tensor<16xi32>, tensor<16xf64>
// CHECK-NEXT:  }
