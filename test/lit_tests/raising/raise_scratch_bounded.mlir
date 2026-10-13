// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// The lane extent has no clamp or guard, but every lane stores into a static
// 16-element scratch buffer at its own index, so the axis cannot exceed 16:
// it batches at 16 behind an iv < extent guard and the barrier raises.
func.func @scratch_bounded(%out: memref<32xf64, 1>, %in: memref<32xf64, 1>, %n: index) {
  %c1 = arith.constant 1 : index
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %n, %c1, %c1) ({
    affine.parallel (%e) = (0) to (1) {
      %scr = memref.alloca() : memref<16xf64>
      affine.parallel (%t) = (0) to (symbol(%n)) {
        %v = affine.load %in[%t] : memref<32xf64, 1>
        affine.store %v, %scr[%t] : memref<16xf64>
        "enzymexla.barrier"(%t, %c1, %c1) : (index, index, index) -> ()
        %w = affine.load %scr[0] : memref<16xf64>
        affine.store %w, %out[%t] : memref<32xf64, 1>
      }
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK:    func.func @scratch_bounded(%arg0: memref<32xf64, 1>, %arg1: memref<32xf64, 1>, %arg2: index) {
// CHECK-NEXT:    %alloca = memref.alloca() : memref<i64>
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %memref = gpu.alloc  () : memref<i64, 1>
// CHECK-NEXT:    %0 = arith.index_cast %arg2 : index to i64
// CHECK-NEXT:    affine.store %0, %alloca[] : memref<i64>
// CHECK-NEXT:    %c8 = arith.constant 8 : index
// CHECK-NEXT:    enzymexla.memcpy  %memref, %alloca, %c8 : memref<i64, 1>, memref<i64>
// CHECK-NEXT:    enzymexla.xla_wrapper @rxla$raised_0 (%arg1, %arg0, %memref) : (memref<32xf64, 1>, memref<32xf64, 1>, memref<i64, 1>) -> ()
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    gpu.dealloc  %memref : memref<i64, 1>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @rxla$raised_0(%arg0: tensor<32xf64>, %arg1: tensor<32xf64>, %arg2: tensor<i64>) -> (tensor<32xf64>, tensor<32xf64>, tensor<i64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<1xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c_0 : tensor<1xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_1 : tensor<1xi64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x16xf64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %c_2 : tensor<16xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %c_3 : tensor<16xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %6 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<16xi64>
// CHECK-NEXT:    %7 = stablehlo.multiply %5, %6 : tensor<16xi64>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i64>) -> tensor<16xi64>
// CHECK-NEXT:    %9 = stablehlo.add %7, %8 : tensor<16xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %10 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i64>) -> tensor<16xi64>
// CHECK-NEXT:    %11 = stablehlo.add %9, %10 : tensor<16xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %12 = stablehlo.compare GE, %11, %c_6 : (tensor<16xi64>, tensor<16xi64>) -> tensor<16xi1>
// CHECK-NEXT:    %13 = stablehlo.slice %arg0 [0:16] : (tensor<32xf64>) -> tensor<16xf64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_12 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_13 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_14 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_15 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_16 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_17 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_18 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %14 = stablehlo.broadcast_in_dim %13, dims = [1] : (tensor<16xf64>) -> tensor<1x16xf64>
// CHECK-NEXT:    %15 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<16xi1>) -> tensor<16x1xi1>
// CHECK-NEXT:    %16 = stablehlo.broadcast_in_dim %14, dims = [1, 0] : (tensor<1x16xf64>) -> tensor<16x1xf64>
// CHECK-NEXT:    %17 = stablehlo.broadcast_in_dim %cst, dims = [1, 0] : (tensor<1x16xf64>) -> tensor<16x1xf64>
// CHECK-NEXT:    %18 = stablehlo.select %15, %16, %17 : tensor<16x1xi1>, tensor<16x1xf64>
// CHECK-NEXT:    %19 = stablehlo.broadcast_in_dim %18, dims = [1, 0] : (tensor<16x1xf64>) -> tensor<1x16xf64>
// CHECK-NEXT:    %20 = stablehlo.dynamic_update_slice %cst, %19, %c_12, %c_18 : (tensor<1x16xf64>, tensor<1x16xf64>, tensor<i64>, tensor<i64>) -> tensor<1x16xf64>
// CHECK-NEXT:    %21 = stablehlo.slice %20 [0:1, 0:1] : (tensor<1x16xf64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %22 = stablehlo.reshape %21 : (tensor<1x1xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %c_19 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_20 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_21 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_22 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_23 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_24 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %23 = stablehlo.reshape %22 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:    %24 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<f64>) -> tensor<16xf64>
// CHECK-NEXT:    %25 = stablehlo.slice %arg1 [0:16] : (tensor<32xf64>) -> tensor<16xf64>
// CHECK-NEXT:    %26 = stablehlo.select %12, %24, %25 : tensor<16xi1>, tensor<16xf64>
// CHECK-NEXT:    %27 = stablehlo.dynamic_update_slice %arg1, %26, %c_24 : (tensor<32xf64>, tensor<16xf64>, tensor<i64>) -> tensor<32xf64>
// CHECK-NEXT:    return %arg0, %27, %arg2 : tensor<32xf64>, tensor<32xf64>, tensor<i64>
// CHECK-NEXT:  }
