// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --arith-raise --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

module {
  func.func @main(%output: memref<3x3xi64>, %values: memref<3x3xi64>) {
    affine.parallel (%i, %j) = (0, 0) to (3, 3) {
        %val = memref.load %values[%i, %j] : memref<3x3xi64>
        affine.store %val, %output[%i, %j] : memref<3x3xi64>
    }
    return
  }
}
// CHECK:  func.func private @main_raised(%arg0: tensor<3x3xi64>, %arg1: tensor<3x3xi64>) -> (tensor<3x3xi64>, tensor<3x3xi64>) {
// CHECK-NEXT:    %0 = stablehlo.iota dim = 1 : tensor<3x3x1xi64>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 0 : tensor<3x3x1xi64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %1, %0, dim = 2 : (tensor<3x3x1xi64>, tensor<3x3x1xi64>) -> tensor<3x3x2xi64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<3x3x2xi64>) -> tensor<9x2xi64>
// CHECK-NEXT:    %4 = "stablehlo.gather"(%arg1, %3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<3x3xi64>, tensor<9x2xi64>) -> tensor<9xi64>
// CHECK-NEXT:    %5 = stablehlo.reshape %4 : (tensor<9xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    return %5, %arg1 : tensor<3x3xi64>, tensor<3x3xi64>
// CHECK-NEXT:  }

module {
  func.func @single_dim(%output: memref<3xi64>, %values: memref<3xi64>) {
    affine.parallel (%i) = (0) to (3) {
        %val = memref.load %values[%i] : memref<3xi64>
        affine.store %val, %output[%i] : memref<3xi64>
    }
    return
  }
}
// CHECK:  func.func private @single_dim_raised(%arg0: tensor<3xi64>, %arg1: tensor<3xi64>) -> (tensor<3xi64>, tensor<3xi64>) {
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<3x1xi64>
// CHECK-NEXT:    %1 = "stablehlo.gather"(%arg1, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<3xi64>, tensor<3x1xi64>) -> tensor<3xi64>
// CHECK-NEXT:    return %1, %arg1 : tensor<3xi64>, tensor<3xi64>
// CHECK-NEXT:  }

module {
  func.func @with_constant(%output: memref<3x3xi64>, %values: memref<3x3xi64>) {
    %c = arith.constant 0 : index
    affine.parallel (%i) = (0) to (3) {
        %val = memref.load %values[%i, %c] : memref<3x3xi64>
        affine.store %val, %output[%i, %c] : memref<3x3xi64>
    }
    return
  }
}
// CHECK:  func.func private @with_constant_raised(%arg0: tensor<3x3xi64>, %arg1: tensor<3x3xi64>) -> (tensor<3x3xi64>, tensor<3x3xi64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<3x1xi64>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %c, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<3x1xi64>, tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %2 = "stablehlo.gather"(%arg1, %1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<3x3xi64>, tensor<3x2xi64>) -> tensor<3xi64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<3xi64>) -> tensor<3x1xi64>
// CHECK-NEXT:    %4 = stablehlo.slice %arg0 [0:3, 1:3] : (tensor<3x3xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %5 = stablehlo.concatenate %3, %4, dim = 1 : (tensor<3x1xi64>, tensor<3x2xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    return %5, %arg1 : tensor<3x3xi64>, tensor<3x3xi64>
// CHECK-NEXT:  }

module {
  func.func @load_indirect(%output: memref<3x3xi64>, %values: memref<3x3xi64>, %indices: memref<3xi64>) {
    affine.parallel (%i, %j) = (0, 0) to (3, 3) {
      %idx_i64 = affine.load %indices[%j] : memref<3xi64>
      %idx = arith.index_cast %idx_i64 : i64 to index
      %val = memref.load %values[%i, %idx] : memref<3x3xi64>
      affine.store %val, %output[%j, %i] : memref<3x3xi64>
    }
    return
  }
}
// CHECK:  func.func private @load_indirect_raised(%arg0: tensor<3x3xi64>, %arg1: tensor<3x3xi64>, %arg2: tensor<3xi64>) -> (tensor<3x3xi64>, tensor<3x3xi64>, tensor<3xi64>) {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<3xi64>) -> tensor<3x3x1xi64>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 0 : tensor<3x3x1xi64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %1, %0, dim = 2 : (tensor<3x3x1xi64>, tensor<3x3x1xi64>) -> tensor<3x3x2xi64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<3x3x2xi64>) -> tensor<9x2xi64>
// CHECK-NEXT:    %4 = "stablehlo.gather"(%arg1, %3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<3x3xi64>, tensor<9x2xi64>) -> tensor<9xi64>
// CHECK-NEXT:    %5 = stablehlo.reshape %4 {enzymexla.guaranteed_symmetric = false} : (tensor<9xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    %6 = stablehlo.transpose %5, dims = [1, 0] : (tensor<3x3xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    return %6, %arg1, %arg2 : tensor<3x3xi64>, tensor<3x3xi64>, tensor<3xi64>
// CHECK-NEXT:  }

module {
  func.func @repeat_iv(%x: memref<10xi64>, %y: memref<10xi64>, %values: memref<10x10xf64>, %output: memref<10xf64>) {
    affine.parallel (%i) = (0) to (10) {
      %0 = affine.load %x[%i] : memref<10xi64>
      %1 = affine.load %y[%i] : memref<10xi64>
      %xx = arith.index_cast %0 : i64 to index
      %yy = arith.index_cast %1 : i64 to index
      %val = memref.load %values[%yy, %xx] : memref<10x10xf64> // -> tensor<10xf64>
      affine.store %val, %output[%i] : memref<10xf64>
    }
    return
  }
}
// CHECK:  func.func private @repeat_iv_raised(%arg0: tensor<10xi64>, %arg1: tensor<10xi64>, %arg2: tensor<10x10xf64>, %arg3: tensor<10xf64>) -> (tensor<10xi64>, tensor<10xi64>, tensor<10x10xf64>, tensor<10xf64>) {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg1 : (tensor<10xi64>) -> tensor<10x1xi64>
// CHECK-NEXT:    %1 = stablehlo.reshape %arg0 : (tensor<10xi64>) -> tensor<10x1xi64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<10x1xi64>, tensor<10x1xi64>) -> tensor<10x2xi64>
// CHECK-NEXT:    %3 = "stablehlo.gather"(%arg2, %2) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<10x10xf64>, tensor<10x2xi64>) -> tensor<10xf64>
// CHECK-NEXT:    return %arg0, %arg1, %arg2, %3 : tensor<10xi64>, tensor<10xi64>, tensor<10x10xf64>, tensor<10xf64>
// CHECK-NEXT:  }

module {
  func.func @multiple_ivs_per_index_lanes(%x: memref<10x10xi64>, %values: memref<10xf64>, %output: memref<10x10xf64>) {
    affine.parallel (%i, %j) = (0, 0) to (10, 10) {
      %0 = affine.load %x[%i, %j] : memref<10x10xi64> // tensor<10x10>
      %xx = arith.index_cast %0 : i64 to index // tensor<10x10>
      %val = memref.load %values[%xx] : memref<10xf64> // -> tensor<10x10xf64>
      affine.store %val, %output[%i, %j] : memref<10x10xf64> // tensor<10x10>
    }
    return
  }
// CHECK:  func.func private @multiple_ivs_per_index_lanes_raised(%arg0: tensor<10x10xi64>, %arg1: tensor<10xf64>, %arg2: tensor<10x10xf64>) -> (tensor<10x10xi64>, tensor<10xf64>, tensor<10x10xf64>) {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<10x10xi64>) -> tensor<100x1xi64>
// CHECK-NEXT:    %1 = "stablehlo.gather"(%arg1, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<10xf64>, tensor<100x1xi64>) -> tensor<100xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<100xf64>) -> tensor<10x10xf64>
// CHECK-NEXT:    return  %arg0, %arg1, %2 : tensor<10x10xi64>, tensor<10xf64>, tensor<10x10xf64>
// CHECK-NEXT:  }
}
