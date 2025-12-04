// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

module {
  func.func @single_dim(%output: memref<3xi64>, %values: memref<3xi64>) {
    affine.parallel (%i) = (0) to (3) {
        %val = affine.load %values[%i] : memref<3xi64>
        memref.store %val, %output[%i] : memref<3xi64>
    }
    return
  }
}
// CHECK:  func.func private @single_dim_raised(%arg0: tensor<3xi64>, %arg1: tensor<3xi64>) -> (tensor<3xi64>, tensor<3xi64>) {
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<3x1xi64>
// CHECK-NEXT:    %1 = "stablehlo.scatter"(%arg0, %0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<i64>
// CHECK-NEXT:    }) : (tensor<3xi64>, tensor<3x1xi64>, tensor<3xi64>) -> tensor<3xi64>
// CHECK-NEXT:    return %1, %arg1 : tensor<3xi64>, tensor<3xi64>
// CHECK-NEXT:  }

module {
  func.func @multiple_dims(%output: memref<3x3xi64>, %values: memref<3x3xi64>) {
    affine.parallel (%i, %j) = (0, 0) to (3, 3) {
        %val = affine.load %values[%i, %j] : memref<3x3xi64>
        memref.store %val, %output[%i, %j] : memref<3x3xi64>
    }
    return
  }
}
// CHECK:  func.func private @multiple_dims_raised(%arg0: tensor<3x3xi64>, %arg1: tensor<3x3xi64>) -> (tensor<3x3xi64>, tensor<3x3xi64>) {
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<3x3x1xi64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<3x3x1xi64>) -> tensor<9x1xi64>
// CHECK-NEXT:    %2 = stablehlo.iota dim = 1 : tensor<3x3xi64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<3x3xi64>) -> tensor<9x1xi64>
// CHECK-NEXT:    %4 = stablehlo.concatenate %1, %3, dim = 1 : (tensor<9x1xi64>, tensor<9x1xi64>) -> tensor<9x2xi64>
// CHECK-NEXT:    %5 = stablehlo.reshape %arg1 : (tensor<3x3xi64>) -> tensor<9xi64>
// CHECK-NEXT:    %6 = "stablehlo.scatter"(%arg0, %4, %5) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<i64>
// CHECK-NEXT:    }) : (tensor<3x3xi64>, tensor<9x2xi64>, tensor<9xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    return %6, %arg1 : tensor<3x3xi64>, tensor<3x3xi64>
// CHECK-NEXT:  }

module {
  func.func @with_constant(%output: memref<3x3xi64>, %values: memref<3x3xi64>) {
    %c = arith.constant 0 : index
    affine.parallel (%i) = (0) to (3) {
        %val = affine.load %values[%i, %c] : memref<3x3xi64>
        memref.store %val, %output[%i, %c] : memref<3x3xi64>
    }
    return
  }
}
// CHECK:  func.func private @with_constant_raised(%arg0: tensor<3x3xi64>, %arg1: tensor<3x3xi64>) -> (tensor<3x3xi64>, tensor<3x3xi64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:3, 0:1] : (tensor<3x3xi64>) -> tensor<3x1xi64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<3x1xi64>) -> tensor<3xi64>
// CHECK-NEXT:    %2 = stablehlo.iota dim = 0 : tensor<3x1xi64>
// CHECK-NEXT:    %3 = stablehlo.pad %2, %c, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<3x1xi64>, tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %4 = "stablehlo.scatter"(%arg0, %3, %1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<i64>
// CHECK-NEXT:    }) : (tensor<3x3xi64>, tensor<3x2xi64>, tensor<3xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    return %4, %arg1 : tensor<3x3xi64>, tensor<3x3xi64>
// CHECK-NEXT:  }

module {
  func.func @load_indirect(%output: memref<3x3xi64>, %values: memref<3x3xi64>, %indices: memref<3xi64>) {
    affine.parallel (%i, %j) = (0, 0) to (3, 3) {
      %idx_i64 = affine.load %indices[%j] : memref<3xi64>
      %idx = arith.index_cast %idx_i64 : i64 to index
      %val = affine.load %values[%i, %j] : memref<3x3xi64>
      memref.store %val, %output[%idx, %i] : memref<3x3xi64>
    }
    return
  }
}
// CHECK:  func.func private @load_indirect_raised(%arg0: tensor<3x3xi64>, %arg1: tensor<3x3xi64>, %arg2: tensor<3xi64>) -> (tensor<3x3xi64>, tensor<3x3xi64>, tensor<3xi64>) {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<3xi64>) -> tensor<3x3x1xi64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<3x3x1xi64>) -> tensor<9x1xi64>
// CHECK-NEXT:    %2 = stablehlo.iota dim = 1 : tensor<3x3xi64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<3x3xi64>) -> tensor<9x1xi64>
// CHECK-NEXT:    %4 = stablehlo.concatenate %1, %3, dim = 1 : (tensor<9x1xi64>, tensor<9x1xi64>) -> tensor<9x2xi64>
// CHECK-NEXT:    %5 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x3xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    %6 = stablehlo.reshape %5 : (tensor<3x3xi64>) -> tensor<9xi64>
// CHECK-NEXT:    %7 = "stablehlo.scatter"(%arg0, %4, %6) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg3: tensor<i64>, %arg4: tensor<i64>):
// CHECK-NEXT:      stablehlo.return %arg4 : tensor<i64>
// CHECK-NEXT:    }) : (tensor<3x3xi64>, tensor<9x2xi64>, tensor<9xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    return %7, %arg1, %arg2 : tensor<3x3xi64>, tensor<3x3xi64>, tensor<3xi64>
// CHECK-NEXT:  }
