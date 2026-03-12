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
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<3x3x2xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<3x3x2xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<2> : tensor<3x1xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<3x1xi64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<3x1xi64>
// CHECK-NEXT:    %1 = stablehlo.maximum %0, %c_2 : tensor<3x1xi64>
// CHECK-NEXT:    %2 = stablehlo.minimum %1, %c_1 : tensor<3x1xi64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 1 : tensor<3x3x1xi64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %2, dims = [0, 2] : (tensor<3x1xi64>) -> tensor<3x3x1xi64>
// CHECK-NEXT:    %5 = stablehlo.concatenate %4, %3, dim = 2 : (tensor<3x3x1xi64>, tensor<3x3x1xi64>) -> tensor<3x3x2xi64>
// CHECK-NEXT:    %6 = stablehlo.maximum %5, %c_0 : tensor<3x3x2xi64>
// CHECK-NEXT:    %7 = stablehlo.minimum %6, %c : tensor<3x3x2xi64>
// CHECK-NEXT:    %8 = stablehlo.reshape %7 : (tensor<3x3x2xi64>) -> tensor<9x2xi64>
// CHECK-NEXT:    %9 = "stablehlo.gather"(%arg1, %8) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<3x3xi64>, tensor<9x2xi64>) -> tensor<9xi64>
// CHECK-NEXT:    %10 = stablehlo.reshape %9 : (tensor<9xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    return %10, %arg1 : tensor<3x3xi64>, tensor<3x3xi64>
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
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<3x1xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<3x1xi64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<3x1xi64>
// CHECK-NEXT:    %1 = stablehlo.maximum %0, %c_0 : tensor<3x1xi64>
// CHECK-NEXT:    %2 = stablehlo.minimum %1, %c : tensor<3x1xi64>
// CHECK-NEXT:    %3 = "stablehlo.gather"(%arg1, %2) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<3xi64>, tensor<3x1xi64>) -> tensor<3xi64>
// CHECK-NEXT:    return %3, %arg1 : tensor<3xi64>, tensor<3xi64>
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
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<2> : tensor<3x2xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<3x2xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<2> : tensor<3x1xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<3x1xi64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<3x1xi64>
// CHECK-NEXT:    %1 = stablehlo.maximum %0, %c_3 : tensor<3x1xi64>
// CHECK-NEXT:    %2 = stablehlo.minimum %1, %c_2 : tensor<3x1xi64>
// CHECK-NEXT:    %3 = stablehlo.pad %2, %c, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<3x1xi64>, tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %4 = stablehlo.maximum %3, %c_1 : tensor<3x2xi64>
// CHECK-NEXT:    %5 = stablehlo.minimum %4, %c_0 : tensor<3x2xi64>
// CHECK-NEXT:    %6 = "stablehlo.gather"(%arg1, %5) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<3x3xi64>, tensor<3x2xi64>) -> tensor<3xi64>
// CHECK-NEXT:    %7 = stablehlo.reshape %6 : (tensor<3xi64>) -> tensor<3x1xi64>
// CHECK-NEXT:    %8 = stablehlo.slice %arg0 [0:3, 1:3] : (tensor<3x3xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %9 = stablehlo.concatenate %7, %8, dim = 1 : (tensor<3x1xi64>, tensor<3x2xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    return %9, %arg1 : tensor<3x3xi64>, tensor<3x3xi64>
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
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<3x3x2xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<3x3x2xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<2> : tensor<3x1xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<3x1xi64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<3x1xi64>
// CHECK-NEXT:    %1 = stablehlo.maximum %0, %c_2 : tensor<3x1xi64>
// CHECK-NEXT:    %2 = stablehlo.minimum %1, %c_1 : tensor<3x1xi64>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<3xi64>) -> tensor<3x3x1xi64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %2, dims = [0, 2] : (tensor<3x1xi64>) -> tensor<3x3x1xi64>
// CHECK-NEXT:    %5 = stablehlo.concatenate %4, %3, dim = 2 : (tensor<3x3x1xi64>, tensor<3x3x1xi64>) -> tensor<3x3x2xi64>
// CHECK-NEXT:    %6 = stablehlo.maximum %5, %c_0 : tensor<3x3x2xi64>
// CHECK-NEXT:    %7 = stablehlo.minimum %6, %c : tensor<3x3x2xi64>
// CHECK-NEXT:    %8 = stablehlo.reshape %7 : (tensor<3x3x2xi64>) -> tensor<9x2xi64>
// CHECK-NEXT:    %9 = "stablehlo.gather"(%arg1, %8) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<3x3xi64>, tensor<9x2xi64>) -> tensor<9xi64>
// CHECK-NEXT:    %10 = stablehlo.reshape %9 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<9xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    %11 = stablehlo.transpose %10, dims = [1, 0] : (tensor<3x3xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    return %11, %arg1, %arg2 : tensor<3x3xi64>, tensor<3x3xi64>, tensor<3xi64>
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
// CHECK-NEXT:    %c = stablehlo.constant dense<9> : tensor<10x2xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<10x2xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<9> : tensor<10x1xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<10x1xi64>
// CHECK-NEXT:    %0 = stablehlo.reshape %arg1 : (tensor<10xi64>) -> tensor<10x1xi64>
// CHECK-NEXT:    %1 = stablehlo.maximum %0, %c_2 : tensor<10x1xi64>
// CHECK-NEXT:    %2 = stablehlo.minimum %1, %c_1 : tensor<10x1xi64>
// CHECK-NEXT:    %3 = stablehlo.reshape %arg0 : (tensor<10xi64>) -> tensor<10x1xi64>
// CHECK-NEXT:    %4 = stablehlo.concatenate %2, %3, dim = 1 : (tensor<10x1xi64>, tensor<10x1xi64>) -> tensor<10x2xi64>
// CHECK-NEXT:    %5 = stablehlo.maximum %4, %c_0 : tensor<10x2xi64>
// CHECK-NEXT:    %6 = stablehlo.minimum %5, %c : tensor<10x2xi64>
// CHECK-NEXT:    %7 = "stablehlo.gather"(%arg2, %6) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<10x10xf64>, tensor<10x2xi64>) -> tensor<10xf64>
// CHECK-NEXT:    return %arg0, %arg1, %arg2, %7 : tensor<10xi64>, tensor<10xi64>, tensor<10x10xf64>, tensor<10xf64>
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
}
// CHECK:  func.func private @multiple_ivs_per_index_lanes_raised(%arg0: tensor<10x10xi64>, %arg1: tensor<10xf64>, %arg2: tensor<10x10xf64>) -> (tensor<10x10xi64>, tensor<10xf64>, tensor<10x10xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<9> : tensor<100x1xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<100x1xi64>
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<10x10xi64>) -> tensor<100x1xi64>
// CHECK-NEXT:    %1 = stablehlo.maximum %0, %c_0 : tensor<100x1xi64>
// CHECK-NEXT:    %2 = stablehlo.minimum %1, %c : tensor<100x1xi64>
// CHECK-NEXT:    %3 = "stablehlo.gather"(%arg1, %2) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<10xf64>, tensor<100x1xi64>) -> tensor<100xf64>
// CHECK-NEXT:    %4 = stablehlo.reshape %3 : (tensor<100xf64>) -> tensor<10x10xf64>
// CHECK-NEXT:    return %arg0, %arg1, %4 : tensor<10x10xi64>, tensor<10xf64>, tensor<10x10xf64>
// CHECK-NEXT:  }
