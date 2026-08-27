// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

// A memref.store guarded by an affine.if raises to a masked scatter: a
// masked-out lane must not write at all — its index expression is
// unconstrained and can collide with a live lane's slot, and scatter
// applies duplicate indices in unspecified order. Dead lanes' indices are
// selected out of bounds, so the scatter drops their updates.

module {
  func.func @main(%arg0: memref<100xf32>, %arg1: memref<100xf32>) {
    affine.parallel (%i, %j) = (0, 0) to (10, 10) step (1, 1) {
      affine.if affine_set<(d0, d1) : (d0 - d1 >= 0)>(%i, %j) {
        %0 = affine.load %arg1[%i * 10 + %j] : memref<100xf32>
        affine.store %0, %arg0[%i * 10 + %j] : memref<100xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func private @main_raised(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<100xf32>, %[[ARG1:.+]]: tensor<100xf32>
// CHECK-DAG:   %[[OOB:.+]] = stablehlo.constant dense<-1> : tensor<10x10x1xi64>
// CHECK:       %[[MASK:.+]] = stablehlo.compare GE, %{{.+}}, %{{.+}} : (tensor<10x10xi64>, tensor<10x10xi64>) -> tensor<10x10xi1>
// CHECK-NOT:   stablehlo.gather
// CHECK:       %[[MASK3:.+]] = stablehlo.reshape %[[MASK]] : (tensor<10x10xi1>) -> tensor<10x10x1xi1>
// CHECK:       %[[IDX:.+]] = stablehlo.select %[[MASK3]], %{{.+}}, %[[OOB]] : tensor<10x10x1xi1>, tensor<10x10x1xi64>
// CHECK:       %[[SCATTER:.+]] = "stablehlo.scatter"(%[[ARG0]], %[[IDX]], %{{.+}}) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}>
// CHECK:         stablehlo.return %{{.+}} : tensor<f32>
// CHECK:       return %[[SCATTER]], %[[ARG1]] : tensor<100xf32>, tensor<100xf32>
