// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// An atomic add whose result is unused raises as a combining scatter: adds
// commute (up to rounding, exactly like the atomic), so application order
// is irrelevant even at colliding indices. The llvm.atomicrmw form arrives
// here as memref.atomic_rmw via llvm-to-memref-access.

module {
  func.func private @atomic_add(%buf: memref<8xf64, 1>, %idx: memref<16xi32, 1>, %val: memref<16xf64, 1>) {
    affine.parallel (%t) = (0) to (16) {
      %i = affine.load %idx[%t] : memref<16xi32, 1>
      %iidx = arith.index_cast %i : i32 to index
      %v = affine.load %val[%t] : memref<16xf64, 1>
      %old = memref.atomic_rmw addf %v, %buf[%iidx] : (f64, memref<8xf64, 1>) -> f64
    }
    return
  }
}

// CHECK:  func.func private @atomic_add_raised(%[[buf:.+]]: tensor<8xf64>, %[[idx:.+]]: tensor<16xi32>, %[[val:.+]]: tensor<16xf64>) -> (tensor<8xf64>, tensor<16xi32>, tensor<16xf64>) {
// CHECK-NEXT:    %[[iota:.+]] = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %[[c0:.+]] = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %[[lanes:.+]] = stablehlo.add %[[iota]], %[[c0]] : tensor<16xi64>
// CHECK-NEXT:    %[[c1:.+]] = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %{{.+}} = stablehlo.multiply %[[lanes]], %[[c1]] : tensor<16xi64>
// CHECK-NEXT:    %[[idxr:.+]] = stablehlo.reshape %[[idx]] : (tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT:    %[[idx64:.+]] = stablehlo.convert %[[idxr]] : (tensor<16xi32>) -> tensor<16xi64>
// CHECK-NEXT:    %[[valr:.+]] = stablehlo.reshape %[[val]] : (tensor<16xf64>) -> tensor<16xf64>
// CHECK-NEXT:    %[[sidx:.+]] = stablehlo.reshape %[[idx64]] : (tensor<16xi64>) -> tensor<16x1xi64>
// CHECK-NEXT:    %[[upd:.+]] = stablehlo.broadcast_in_dim %[[valr]], dims = [0] : (tensor<16xf64>) -> tensor<16xf64>
// CHECK-NEXT:    %[[scat:.+]] = "stablehlo.scatter"(%[[buf]], %[[sidx]], %[[upd]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%[[a:.+]]: tensor<f64>, %[[b:.+]]: tensor<f64>):
// CHECK-NEXT:      %[[sum:.+]] = stablehlo.add %[[a]], %[[b]] : tensor<f64>
// CHECK-NEXT:      stablehlo.return %[[sum]] : tensor<f64>
// CHECK-NEXT:    }) : (tensor<8xf64>, tensor<16x1xi64>, tensor<16xf64>) -> tensor<8xf64>
// CHECK-NEXT:    return %[[scat]], %[[idx]], %[[val]] : tensor<8xf64>, tensor<16xi32>, tensor<16xf64>
// CHECK-NEXT:  }

// -----

// Integer atomic adds take the same path.

module {
  func.func private @atomic_add_int(%buf: memref<8xi32, 1>, %idx: memref<16xi32, 1>, %val: memref<16xi32, 1>) {
    affine.parallel (%t) = (0) to (16) {
      %i = affine.load %idx[%t] : memref<16xi32, 1>
      %iidx = arith.index_cast %i : i32 to index
      %v = affine.load %val[%t] : memref<16xi32, 1>
      %old = memref.atomic_rmw addi %v, %buf[%iidx] : (i32, memref<8xi32, 1>) -> i32
    }
    return
  }
}

// CHECK:  func.func private @atomic_add_int_raised(%[[buf:.+]]: tensor<8xi32>, %[[idx:.+]]: tensor<16xi32>, %[[val:.+]]: tensor<16xi32>) -> (tensor<8xi32>, tensor<16xi32>, tensor<16xi32>) {
// CHECK-NEXT:    %[[iota:.+]] = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %[[c0:.+]] = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %[[lanes:.+]] = stablehlo.add %[[iota]], %[[c0]] : tensor<16xi64>
// CHECK-NEXT:    %[[c1:.+]] = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %{{.+}} = stablehlo.multiply %[[lanes]], %[[c1]] : tensor<16xi64>
// CHECK-NEXT:    %[[idxr:.+]] = stablehlo.reshape %[[idx]] : (tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT:    %[[idx64:.+]] = stablehlo.convert %[[idxr]] : (tensor<16xi32>) -> tensor<16xi64>
// CHECK-NEXT:    %[[valr:.+]] = stablehlo.reshape %[[val]] : (tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT:    %[[sidx:.+]] = stablehlo.reshape %[[idx64]] : (tensor<16xi64>) -> tensor<16x1xi64>
// CHECK-NEXT:    %[[upd:.+]] = stablehlo.broadcast_in_dim %[[valr]], dims = [0] : (tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT:    %[[scat:.+]] = "stablehlo.scatter"(%[[buf]], %[[sidx]], %[[upd]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%[[a:.+]]: tensor<i32>, %[[b:.+]]: tensor<i32>):
// CHECK-NEXT:      %[[sum:.+]] = stablehlo.add %[[a]], %[[b]] : tensor<i32>
// CHECK-NEXT:      stablehlo.return %[[sum]] : tensor<i32>
// CHECK-NEXT:    }) : (tensor<8xi32>, tensor<16x1xi64>, tensor<16xi32>) -> tensor<8xi32>
// CHECK-NEXT:    return %[[scat]], %[[idx]], %[[val]] : tensor<8xi32>, tensor<16xi32>, tensor<16xi32>
// CHECK-NEXT:  }
