// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo="prefer_while_raising=true specialize_index_strides=true" | FileCheck %s
// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo="prefer_while_raising=true" | FileCheck %s --check-prefix=BOUNDS
// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo="specialize_index_strides=true" | FileCheck %s --check-prefix=GATHER

// The element loop runs to `ne`, which the runtime specializes as a loop
// bound. Each element's dofs sit `nd` apart, and `nd` reaches the kernel only
// through the access index `e * nd + d`. With specialize_index_strides the
// stride is specialized as well and the index folds to a constant; without
// it only the bound is.
func.func @stride(%out: memref<?xf64, 1>, %ne: i32, %nd: i32, %in: memref<?xf64, 1>) {
  %c1 = arith.constant 1 : index
  %nei = arith.index_cast %ne : i32 to index
  %ndi = arith.index_cast %nd : i32 to index
  %c0_i32 = arith.constant 0 : i32
  %ok = arith.cmpi sgt, %ne, %c0_i32 : i32
  scf.if %ok {
  %0 = "enzymexla.gpu_wrapper"(%nei, %c1, %c1, %c1, %c1, %c1) ({
    affine.parallel (%e) = (0) to (symbol(%nei)) {
      affine.for %d = 0 to 3 {
        %v = affine.load %in[%e * symbol(%ndi) + %d] : memref<?xf64, 1>
        %w = arith.mulf %v, %v : f64
        affine.store %w, %out[%e * symbol(%ndi) + %d] : memref<?xf64, 1>
      }
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  }
  return
}

// A fixed number of elements, one dof each, raises with no loop at all: a
// gather whose index is `e * nd` over every element. The stride is
// specialized all the same, and the gather of a constant index is a slice.
func.func @gather_only(%out: memref<?xf64, 1>, %nd: i32, %in: memref<?xf64, 1>) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %ndi = arith.index_cast %nd : i32 to index
  %0 = "enzymexla.gpu_wrapper"(%c8, %c1, %c1, %c1, %c1, %c1) ({
    affine.parallel (%e) = (0) to (8) {
      %v = affine.load %in[%e * symbol(%ndi)] : memref<?xf64, 1>
      %w = arith.mulf %v, %v : f64
      affine.store %w, %out[%e * symbol(%ndi)] : memref<?xf64, 1>
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK:  func.func @stride(%arg0: memref<?xf64, 1>, %arg1: i32, %arg2: i32, %arg3: memref<?xf64, 1>) {
// CHECK-NEXT:   %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:   %alloca_0 = memref.alloca() : memref<i32>
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:   %1 = arith.index_cast %arg2 : i32 to index
// CHECK-NEXT:   %2 = arith.cmpi sgt, %arg1, %c0_i32 : i32
// CHECK-NEXT:   scf.if %2 {
// CHECK-NEXT:     affine.store %arg2, %alloca_0[] : memref<i32>
// CHECK-NEXT:     %c4 = arith.constant 4 : index
// CHECK-NEXT:     affine.store %arg1, %alloca[] : memref<i32>
// CHECK-NEXT:     %c4_1 = arith.constant 4 : index
// CHECK-NEXT:     enzymexla.xla_wrapper @rxla$raised_0 (%arg3, %arg0, %arg2, %arg1) <num_specialized = 2> : (memref<?xf64, 1>, memref<?xf64, 1>, i32, i32) -> ()
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK:  func.func @gather_only(%arg0: memref<?xf64, 1>, %arg1: i32, %arg2: memref<?xf64, 1>) {
// CHECK-NEXT:   %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c8 = arith.constant 8 : index
// CHECK-NEXT:   %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:   affine.store %arg1, %alloca[] : memref<i32>
// CHECK-NEXT:   %c4 = arith.constant 4 : index
// CHECK-NEXT:   enzymexla.xla_wrapper @rxla$raised_1 (%arg2, %arg0, %arg1) <num_specialized = 1> : (memref<?xf64, 1>, memref<?xf64, 1>, i32) -> ()
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   return
// CHECK-NEXT: }

// BOUNDS:  func.func @stride(%arg0: memref<?xf64, 1>, %arg1: i32, %arg2: i32, %arg3: memref<?xf64, 1>) {
// BOUNDS-NEXT:   %alloca = memref.alloca() : memref<i32>
// BOUNDS-NEXT:   %alloca_0 = memref.alloca() : memref<i32>
// BOUNDS-NEXT:   %c0_i32 = arith.constant 0 : i32
// BOUNDS-NEXT:   %c1 = arith.constant 1 : index
// BOUNDS-NEXT:   %0 = arith.index_cast %arg1 : i32 to index
// BOUNDS-NEXT:   %1 = arith.index_cast %arg2 : i32 to index
// BOUNDS-NEXT:   %2 = arith.cmpi sgt, %arg1, %c0_i32 : i32
// BOUNDS-NEXT:   scf.if %2 {
// BOUNDS-NEXT:     %memref = gpu.alloc  () : memref<i32, 1>
// BOUNDS-NEXT:     affine.store %arg2, %alloca_0[] : memref<i32>
// BOUNDS-NEXT:     %c4 = arith.constant 4 : index
// BOUNDS-NEXT:     enzymexla.memcpy  %memref, %alloca_0, %c4 : memref<i32, 1>, memref<i32>
// BOUNDS-NEXT:     affine.store %arg1, %alloca[] : memref<i32>
// BOUNDS-NEXT:     %c4_1 = arith.constant 4 : index
// BOUNDS-NEXT:     enzymexla.xla_wrapper @rxla$raised_0 (%arg3, %memref, %arg0, %arg1) <num_specialized = 1> : (memref<?xf64, 1>, memref<i32, 1>, memref<?xf64, 1>, i32) -> ()
// BOUNDS-NEXT:     %c0 = arith.constant 0 : index
// BOUNDS-NEXT:     gpu.dealloc  %memref : memref<i32, 1>
// BOUNDS-NEXT:   }
// BOUNDS-NEXT:   return
// BOUNDS-NEXT: }

// BOUNDS:  func.func @gather_only(%arg0: memref<?xf64, 1>, %arg1: i32, %arg2: memref<?xf64, 1>) {
// BOUNDS-NEXT:   %alloca = memref.alloca() : memref<i32>
// BOUNDS-NEXT:   %c1 = arith.constant 1 : index
// BOUNDS-NEXT:   %c8 = arith.constant 8 : index
// BOUNDS-NEXT:   %0 = arith.index_cast %arg1 : i32 to index
// BOUNDS-NEXT:   %memref = gpu.alloc  () : memref<i32, 1>
// BOUNDS-NEXT:   affine.store %arg1, %alloca[] : memref<i32>
// BOUNDS-NEXT:   %c4 = arith.constant 4 : index
// BOUNDS-NEXT:   enzymexla.memcpy  %memref, %alloca, %c4 : memref<i32, 1>, memref<i32>
// BOUNDS-NEXT:   enzymexla.xla_wrapper @rxla$raised_1 (%arg2, %memref, %arg0) : (memref<?xf64, 1>, memref<i32, 1>, memref<?xf64, 1>) -> ()
// BOUNDS-NEXT:   %c0 = arith.constant 0 : index
// BOUNDS-NEXT:   gpu.dealloc  %memref : memref<i32, 1>
// BOUNDS-NEXT:   return
// BOUNDS-NEXT: }

// GATHER:  func.func @stride(%arg0: memref<?xf64, 1>, %arg1: i32, %arg2: i32, %arg3: memref<?xf64, 1>) {
// GATHER-NEXT:   %alloca = memref.alloca() : memref<i32>
// GATHER-NEXT:   %alloca_0 = memref.alloca() : memref<i32>
// GATHER-NEXT:   %c0_i32 = arith.constant 0 : i32
// GATHER-NEXT:   %c1 = arith.constant 1 : index
// GATHER-NEXT:   %0 = arith.index_cast %arg1 : i32 to index
// GATHER-NEXT:   %1 = arith.index_cast %arg2 : i32 to index
// GATHER-NEXT:   %2 = arith.cmpi sgt, %arg1, %c0_i32 : i32
// GATHER-NEXT:   scf.if %2 {
// GATHER-NEXT:     affine.store %arg2, %alloca_0[] : memref<i32>
// GATHER-NEXT:     %c4 = arith.constant 4 : index
// GATHER-NEXT:     affine.store %arg1, %alloca[] : memref<i32>
// GATHER-NEXT:     %c4_1 = arith.constant 4 : index
// GATHER-NEXT:     enzymexla.xla_wrapper @rxla$raised_0 (%arg3, %arg0, %arg2, %arg1) <num_specialized = 2> : (memref<?xf64, 1>, memref<?xf64, 1>, i32, i32) -> ()
// GATHER-NEXT:     %c0 = arith.constant 0 : index
// GATHER-NEXT:   }
// GATHER-NEXT:   return
// GATHER-NEXT: }

// GATHER:  func.func @gather_only(%arg0: memref<?xf64, 1>, %arg1: i32, %arg2: memref<?xf64, 1>) {
// GATHER-NEXT:   %alloca = memref.alloca() : memref<i32>
// GATHER-NEXT:   %c1 = arith.constant 1 : index
// GATHER-NEXT:   %c8 = arith.constant 8 : index
// GATHER-NEXT:   %0 = arith.index_cast %arg1 : i32 to index
// GATHER-NEXT:   affine.store %arg1, %alloca[] : memref<i32>
// GATHER-NEXT:   %c4 = arith.constant 4 : index
// GATHER-NEXT:   enzymexla.xla_wrapper @rxla$raised_1 (%arg2, %arg0, %arg1) <num_specialized = 1> : (memref<?xf64, 1>, memref<?xf64, 1>, i32) -> ()
// GATHER-NEXT:   %c0 = arith.constant 0 : index
// GATHER-NEXT:   return
// GATHER-NEXT: }

// GATHER:  func.func private @rxla$raised_1(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<i32>) -> (tensor<?xf64>, tensor<?xf64>) {
// GATHER-NEXT:   %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<i64>
// GATHER-NEXT:   %1 = stablehlo.iota dim = 0 : tensor<8xi64>
// GATHER-NEXT:   %c = stablehlo.constant dense<0> : tensor<8xi64>
// GATHER-NEXT:   %2 = stablehlo.add %1, %c : tensor<8xi64>
// GATHER-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<8xi64>
// GATHER-NEXT:   %3 = stablehlo.multiply %2, %c_0 : tensor<8xi64>
// GATHER-NEXT:   %4 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i64>) -> tensor<8xi64>
// GATHER-NEXT:   %5 = stablehlo.multiply %3, %4 : tensor<8xi64>
// GATHER-NEXT:   %6 = stablehlo.reshape %5 : (tensor<8xi64>) -> tensor<8x1xi64>
// GATHER-NEXT:   %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<?xf64>, tensor<8x1xi64>) -> tensor<8xf64>
// GATHER-NEXT:   %8 = arith.mulf %7, %7 : tensor<8xf64>
// GATHER-NEXT:   %9 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i64>) -> tensor<8xi64>
// GATHER-NEXT:   %10 = stablehlo.multiply %3, %9 : tensor<8xi64>
// GATHER-NEXT:   %11 = stablehlo.reshape %10 : (tensor<8xi64>) -> tensor<8x1xi64>
// GATHER-NEXT:   %12 = "stablehlo.scatter"(%arg1, %11, %8) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// GATHER-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// GATHER-NEXT:     stablehlo.return %arg4 : tensor<f64>
// GATHER-NEXT:   }) : (tensor<?xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<?xf64>
// GATHER-NEXT:   return %arg0, %12 : tensor<?xf64>, tensor<?xf64>
// GATHER-NEXT: }
