// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// Two lane axes share the block extent Q: a block of Q*Q threads never
// launches past 1024, so each axis is bounded by 32 and pads to that behind
// an iv < Q guard; the barrier between the scratch fill and the read raises.
func.func @budget(%out: memref<1024xf64, 1>, %in: memref<1024xf64, 1>, %qbuf: memref<i32, 1>, %unused: index) {
  %c1 = arith.constant 1 : index
  %q = affine.load %qbuf[] : memref<i32, 1>
  %qi = arith.index_cast %q : i32 to index
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %qi, %qi, %c1) ({
    affine.parallel (%e) = (0) to (1) {
      %scr = memref.alloca() : memref<32x32xf64>
      affine.parallel (%tx, %ty) = (0, 0) to (symbol(%qi), symbol(%qi)) {
        %v = affine.load %in[%tx * 32 + %ty] : memref<1024xf64, 1>
        affine.store %v, %scr[%tx, %ty] : memref<32x32xf64>
        "enzymexla.barrier"(%tx, %ty, %c1) : (index, index, index) -> ()
        %w = affine.load %scr[%ty, %tx] : memref<32x32xf64>
        affine.store %w, %out[%tx * 32 + %ty] : memref<1024xf64, 1>
      }
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK:      func.func @budget(%arg0: memref<1024xf64, 1>, %arg1: memref<1024xf64, 1>, %arg2: memref<i32, 1>, %arg3: index) {
// CHECK-NEXT:    %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %0 = affine.load %arg2[] : memref<i32, 1>
// CHECK-NEXT:    %1 = arith.index_cast %0 : i32 to index
// CHECK-NEXT:    %memref = gpu.alloc  () : memref<i32, 1>
// CHECK-NEXT:    affine.store %0, %alloca[] : memref<i32>
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    enzymexla.memcpy  %memref, %alloca, %c4 : memref<i32, 1>, memref<i32>
// CHECK-NEXT:    enzymexla.xla_wrapper @rxla$raised_0 (%arg1, %arg0, %memref) : (memref<1024xf64, 1>, memref<1024xf64, 1>, memref<i32, 1>) -> ()
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    gpu.dealloc  %memref : memref<i32, 1>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @rxla$raised_0(%arg0: tensor<1024xf64>, %arg1: tensor<1024xf64>, %arg2: tensor<i32>) -> (tensor<1024xf64>, tensor<1024xf64>, tensor<i32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 0 : tensor<1xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %2 = stablehlo.add %1, %c_0 : tensor<1xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NEXT:    %3 = stablehlo.multiply %2, %c_1 : tensor<1xi64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x32x32xf64>
// CHECK-NEXT:    %4 = stablehlo.iota dim = 0 : tensor<32xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<32xi64>
// CHECK-NEXT:    %5 = stablehlo.add %4, %c_2 : tensor<32xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<1> : tensor<32xi64>
// CHECK-NEXT:    %6 = stablehlo.multiply %5, %c_3 : tensor<32xi64>
// CHECK-NEXT:    %7 = stablehlo.iota dim = 0 : tensor<32xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<32xi64>
// CHECK-NEXT:    %8 = stablehlo.add %7, %c_4 : tensor<32xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<1> : tensor<32xi64>
// CHECK-NEXT:    %9 = stablehlo.multiply %8, %c_5 : tensor<32xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %10 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i64>) -> tensor<32xi64>
// CHECK-NEXT:    %11 = stablehlo.multiply %6, %10 : tensor<32xi64>
// CHECK-NEXT:    %12 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i64>) -> tensor<32xi64>
// CHECK-NEXT:    %13 = stablehlo.add %11, %12 : tensor<32xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %14 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i64>) -> tensor<32xi64>
// CHECK-NEXT:    %15 = stablehlo.add %13, %14 : tensor<32xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<32xi64>
// CHECK-NEXT:    %16 = stablehlo.compare GE, %15, %c_8 : (tensor<32xi64>, tensor<32xi64>) -> tensor<32xi1>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %17 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i64>) -> tensor<32xi64>
// CHECK-NEXT:    %18 = stablehlo.multiply %9, %17 : tensor<32xi64>
// CHECK-NEXT:    %19 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i64>) -> tensor<32xi64>
// CHECK-NEXT:    %20 = stablehlo.add %18, %19 : tensor<32xi64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %21 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i64>) -> tensor<32xi64>
// CHECK-NEXT:    %22 = stablehlo.add %20, %21 : tensor<32xi64>
// CHECK-NEXT:    %c_11 = stablehlo.constant dense<0> : tensor<32xi64>
// CHECK-NEXT:    %23 = stablehlo.compare GE, %22, %c_11 : (tensor<32xi64>, tensor<32xi64>) -> tensor<32xi1>
// CHECK-NEXT:    %24 = stablehlo.broadcast_in_dim %16, dims = [0] : (tensor<32xi1>) -> tensor<32x32xi1>
// CHECK-NEXT:    %25 = stablehlo.broadcast_in_dim %23, dims = [1] : (tensor<32xi1>) -> tensor<32x32xi1>
// CHECK-NEXT:    %26 = stablehlo.and %24, %25 : tensor<32x32xi1>
// CHECK-NEXT:    %c_12 = stablehlo.constant dense<32> : tensor<i64>
// CHECK-NEXT:    %27 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i64>) -> tensor<32xi64>
// CHECK-NEXT:    %28 = stablehlo.multiply %6, %27 : tensor<32xi64>
// CHECK-NEXT:    %29 = stablehlo.broadcast_in_dim %28, dims = [0] : (tensor<32xi64>) -> tensor<32x32xi64>
// CHECK-NEXT:    %30 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<32xi64>) -> tensor<32x32xi64>
// CHECK-NEXT:    %31 = stablehlo.add %29, %30 : tensor<32x32xi64>
// CHECK-NEXT:    %32 = stablehlo.reshape %31 : (tensor<32x32xi64>) -> tensor<32x32x1xi64>
// CHECK-NEXT:    %33 = "stablehlo.gather"(%arg0, %32) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<1024xf64>, tensor<32x32x1xi64>) -> tensor<32x32xf64>
// CHECK-NEXT:    %c_13 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_14 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_15 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_16 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_17 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_18 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_19 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_20 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_21 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_22 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_23 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_24 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_25 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_26 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_27 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_28 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_29 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_30 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %34 = stablehlo.broadcast_in_dim %33, dims = [1, 2] : (tensor<32x32xf64>) -> tensor<1x32x32xf64>
// CHECK-NEXT:    %35 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<32x32xi1>) -> tensor<32x32x1xi1>
// CHECK-NEXT:    %36 = stablehlo.broadcast_in_dim %34, dims = [2, 0, 1] : (tensor<1x32x32xf64>) -> tensor<32x32x1xf64>
// CHECK-NEXT:    %37 = stablehlo.broadcast_in_dim %cst, dims = [2, 0, 1] : (tensor<1x32x32xf64>) -> tensor<32x32x1xf64>
// CHECK-NEXT:    %38 = stablehlo.select %35, %36, %37 : tensor<32x32x1xi1>, tensor<32x32x1xf64>
// CHECK-NEXT:    %39 = stablehlo.broadcast_in_dim %38, dims = [1, 2, 0] : (tensor<32x32x1xf64>) -> tensor<1x32x32xf64>
// CHECK-NEXT:    %40 = stablehlo.dynamic_update_slice %cst, %39, %c_18, %c_24, %c_30 : (tensor<1x32x32xf64>, tensor<1x32x32xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x32x32xf64>
// CHECK-NEXT:    %c_31 = stablehlo.constant dense<32> : tensor<i64>
// CHECK-NEXT:    %41 = stablehlo.broadcast_in_dim %c_31, dims = [] : (tensor<i64>) -> tensor<32xi64>
// CHECK-NEXT:    %42 = stablehlo.multiply %6, %41 : tensor<32xi64>
// CHECK-NEXT:    %43 = stablehlo.broadcast_in_dim %42, dims = [0] : (tensor<32xi64>) -> tensor<32x32xi64>
// CHECK-NEXT:    %44 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<32xi64>) -> tensor<32x32xi64>
// CHECK-NEXT:    %45 = stablehlo.add %43, %44 : tensor<32x32xi64>
// CHECK-NEXT:    %46 = stablehlo.reshape %45 : (tensor<32x32xi64>) -> tensor<32x32x1xi64>
// CHECK-NEXT:    %47 = stablehlo.reshape %40 : (tensor<1x32x32xf64>) -> tensor<32x32xf64>
// CHECK-NEXT:    %48 = stablehlo.broadcast_in_dim %47, dims = [1, 0] : (tensor<32x32xf64>) -> tensor<32x32xf64>
// CHECK-NEXT:    %c_32 = stablehlo.constant dense<-1> : tensor<32x32x1xi64>
// CHECK-NEXT:    %49 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<32x32xi1>) -> tensor<32x32x1xi1>
// CHECK-NEXT:    %50 = stablehlo.select %49, %46, %c_32 : tensor<32x32x1xi1>, tensor<32x32x1xi64>
// CHECK-NEXT:    %51 = "stablehlo.scatter"(%arg1, %50, %48) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<1024xf64>, tensor<32x32x1xi64>, tensor<32x32xf64>) -> tensor<1024xf64>
// CHECK-NEXT:    return %arg0, %51, %arg2 : tensor<1024xf64>, tensor<1024xf64>, tensor<i32>
// CHECK-NEXT:  }
