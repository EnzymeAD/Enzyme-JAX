// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// The lane extent is a bounded value shifted left (2*min(d, 8) as `shli`):
// the bound shifts with it and the barrier-spanning axis batches at 16.
func.func @shifted(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>, %dbuf: memref<i32, 1>, %unused: index) {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %d = affine.load %dbuf[] : memref<i32, 1>
  %m = arith.minsi %d, %c8 : i32
  %s = arith.shli %m, %c1_i32 : i32
  %ext = arith.index_cast %s : i32 to index
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %ext, %c1, %c1) ({
    affine.parallel (%e) = (0) to (1) {
      %scr = memref.alloca() : memref<16xf64>
      affine.parallel (%t) = (0) to (symbol(%ext)) {
        %v = affine.load %in[%t] : memref<16xf64, 1>
        affine.store %v, %scr[%t] : memref<16xf64>
        "enzymexla.barrier"(%t, %c1, %c1) : (index, index, index) -> ()
        %w = affine.load %scr[0] : memref<16xf64>
        affine.store %w, %out[%t] : memref<16xf64, 1>
      }
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK:    func.func @shifted(%arg0: memref<16xf64, 1>, %arg1: memref<16xf64, 1>, %arg2: memref<i32, 1>, %arg3: index) {
// CHECK-NEXT:    %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:    %0 = affine.load %arg2[] : memref<i32, 1>
// CHECK-NEXT:    %1 = arith.minsi %0, %c8_i32 : i32
// CHECK-NEXT:    %2 = arith.shli %1, %c1_i32 : i32
// CHECK-NEXT:    %3 = arith.index_cast %2 : i32 to index
// CHECK-NEXT:    %memref = gpu.alloc  () : memref<i32, 1>
// CHECK-NEXT:    affine.store %2, %alloca[] : memref<i32>
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    enzymexla.memcpy  %memref, %alloca, %c4 : memref<i32, 1>, memref<i32>
// CHECK-NEXT:    enzymexla.xla_wrapper @rxla$raised_0 (%arg1, %arg0, %memref) : (memref<16xf64, 1>, memref<16xf64, 1>, memref<i32, 1>) -> ()
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    gpu.dealloc  %memref : memref<i32, 1>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @rxla$raised_0(%arg0: tensor<16xf64>, %arg1: tensor<16xf64>, %arg2: tensor<i32>) -> (tensor<16xf64>, tensor<16xf64>, tensor<i32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 0 : tensor<1xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %2 = stablehlo.add %1, %c_0 : tensor<1xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NEXT:    %3 = stablehlo.multiply %2, %c_1 : tensor<1xi64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x16xf64>
// CHECK-NEXT:    %4 = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %5 = stablehlo.add %4, %c_2 : tensor<16xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %6 = stablehlo.multiply %5, %c_3 : tensor<16xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %7 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<16xi64>
// CHECK-NEXT:    %8 = stablehlo.multiply %6, %7 : tensor<16xi64>
// CHECK-NEXT:    %9 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i64>) -> tensor<16xi64>
// CHECK-NEXT:    %10 = stablehlo.add %8, %9 : tensor<16xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %11 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i64>) -> tensor<16xi64>
// CHECK-NEXT:    %12 = stablehlo.add %10, %11 : tensor<16xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %13 = stablehlo.compare GE, %12, %c_6 : (tensor<16xi64>, tensor<16xi64>) -> tensor<16xi1>
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
// CHECK-NEXT:    %14 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<16xf64>) -> tensor<1x16xf64>
// CHECK-NEXT:    %15 = stablehlo.broadcast_in_dim %13, dims = [0] : (tensor<16xi1>) -> tensor<16x1xi1>
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
// CHECK-NEXT:    %25 = stablehlo.select %13, %24, %arg1 : tensor<16xi1>, tensor<16xf64>
// CHECK-NEXT:    %26 = stablehlo.dynamic_update_slice %arg1, %25, %c_24 : (tensor<16xf64>, tensor<16xf64>, tensor<i64>) -> tensor<16xf64>
// CHECK-NEXT:    return %arg0, %26, %arg2 : tensor<16xf64>, tensor<16xf64>, tensor<i32>
// CHECK-NEXT:  }
