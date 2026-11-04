// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s
// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo="specialize_loop_bounds=false" | FileCheck %s --check-prefix=PLAIN

// The launch's block extent `n` is a scalar the host computes: the lanes it
// bounds are batched at the launch budget behind an `iv < n` guard. With
// specialize_loop_bounds it is specialized like an element count: the
// wrapper counts it among the scalars the runtime specializes the
// executable over, so the guard folds to a constant mask and the padding
// can be cut.
func.func @launch_dim(%out: memref<256xf64, 1>, %in: memref<256xf64, 1>, %nbuf: memref<i32, 1>, %unused: index) {
  %c1 = arith.constant 1 : index
  %n = affine.load %nbuf[] : memref<i32, 1>
  %bi = arith.index_cast %n : i32 to index
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %bi, %c1, %c1) ({
    affine.parallel (%e) = (0) to (1) {
      affine.parallel (%t) = (0) to (symbol(%bi)) {
        %v = affine.load %in[%t] : memref<256xf64, 1>
        %w = arith.mulf %v, %v : f64
        affine.store %w, %out[%t] : memref<256xf64, 1>
      }
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK:      func.func @launch_dim(%arg0: memref<256xf64, 1>, %arg1: memref<256xf64, 1>, %arg2: memref<i32, 1>, %arg3: index) {
// CHECK-NEXT:   %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %0 = affine.load %arg2[] : memref<i32, 1>
// CHECK-NEXT:   %1 = arith.index_cast %0 : i32 to index
// CHECK-NEXT:   affine.store %0, %alloca[] : memref<i32>
// CHECK-NEXT:   %c4 = arith.constant 4 : index
// CHECK-NEXT:   enzymexla.xla_wrapper @rxla$raised_0 (%arg1, %arg0, %0) <num_specialized = 1> : (memref<256xf64, 1>, memref<256xf64, 1>, i32) -> ()
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   return
// CHECK-NEXT: }
// CHECK-NEXT: func.func private @rxla$raised_0(%arg0: tensor<256xf64>, %arg1: tensor<256xf64>, %arg2: tensor<i32>) -> (tensor<256xf64>, tensor<256xf64>) {
// CHECK-NEXT:   %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:   %1 = stablehlo.iota dim = 0 : tensor<1xi64>
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:   %2 = stablehlo.add %1, %c : tensor<1xi64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NEXT:   %3 = stablehlo.multiply %2, %c_0 : tensor<1xi64>
// CHECK-NEXT:   %4 = stablehlo.iota dim = 0 : tensor<1024xi64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<0> : tensor<1024xi64>
// CHECK-NEXT:   %5 = stablehlo.add %4, %c_1 : tensor<1024xi64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<1> : tensor<1024xi64>
// CHECK-NEXT:   %6 = stablehlo.multiply %5, %c_2 : tensor<1024xi64>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:   %7 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<1024xi64>
// CHECK-NEXT:   %8 = stablehlo.multiply %6, %7 : tensor<1024xi64>
// CHECK-NEXT:   %9 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i64>) -> tensor<1024xi64>
// CHECK-NEXT:   %10 = stablehlo.add %8, %9 : tensor<1024xi64>
// CHECK-NEXT:   %c_4 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:   %11 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<1024xi64>
// CHECK-NEXT:   %12 = stablehlo.add %10, %11 : tensor<1024xi64>
// CHECK-NEXT:   %c_5 = stablehlo.constant dense<0> : tensor<1024xi64>
// CHECK-NEXT:   %13 = stablehlo.compare GE, %12, %c_5 : (tensor<1024xi64>, tensor<1024xi64>) -> tensor<1024xi1>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %14 = stablehlo.pad %arg0, %cst, low = [0], high = [768], interior = [0] : (tensor<256xf64>, tensor<f64>) -> tensor<1024xf64>
// CHECK-NEXT:   %15 = arith.mulf %14, %14 : tensor<1024xf64>
// CHECK-NEXT:   %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:   %c_7 = stablehlo.constant dense<768> : tensor<1xi64>
// CHECK-NEXT:   %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:   %c_9 = stablehlo.constant dense<-768> : tensor<1xi64>
// CHECK-NEXT:   %c_10 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:   %c_11 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %16 = stablehlo.pad %arg1, %cst_12, low = [0], high = [768], interior = [0] : (tensor<256xf64>, tensor<f64>) -> tensor<1024xf64>
// CHECK-NEXT:   %17 = stablehlo.select %13, %15, %16 : tensor<1024xi1>, tensor<1024xf64>
// CHECK-NEXT:   %18 = stablehlo.dynamic_update_slice %16, %17, %c_11 : (tensor<1024xf64>, tensor<1024xf64>, tensor<i64>) -> tensor<1024xf64>
// CHECK-NEXT:   %19 = stablehlo.slice %18 [0:256] : (tensor<1024xf64>) -> tensor<256xf64>
// CHECK-NEXT:   return %arg0, %19 : tensor<256xf64>, tensor<256xf64>
// CHECK-NEXT: }
// CHECK-NEXT: }

// PLAIN:      func.func @launch_dim(%arg0: memref<256xf64, 1>, %arg1: memref<256xf64, 1>, %arg2: memref<i32, 1>, %arg3: index) {
// PLAIN-NEXT:   %alloca = memref.alloca() : memref<i32>
// PLAIN-NEXT:   %c1 = arith.constant 1 : index
// PLAIN-NEXT:   %0 = affine.load %arg2[] : memref<i32, 1>
// PLAIN-NEXT:   %1 = arith.index_cast %0 : i32 to index
// PLAIN-NEXT:   %memref = gpu.alloc  () : memref<i32, 1>
// PLAIN-NEXT:   affine.store %0, %alloca[] : memref<i32>
// PLAIN-NEXT:   %c4 = arith.constant 4 : index
// PLAIN-NEXT:   enzymexla.memcpy  %memref, %alloca, %c4 : memref<i32, 1>, memref<i32>
// PLAIN-NEXT:   enzymexla.xla_wrapper @rxla$raised_0 (%arg1, %arg0, %memref) : (memref<256xf64, 1>, memref<256xf64, 1>, memref<i32, 1>) -> ()
// PLAIN-NEXT:   %c0 = arith.constant 0 : index
// PLAIN-NEXT:   gpu.dealloc  %memref : memref<i32, 1>
// PLAIN-NEXT:   return
// PLAIN-NEXT: }
// PLAIN-NEXT: func.func private @rxla$raised_0(%arg0: tensor<256xf64>, %arg1: tensor<256xf64>, %arg2: tensor<i32>) -> (tensor<256xf64>, tensor<256xf64>, tensor<i32>) {
// PLAIN-NEXT:   %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<i64>
// PLAIN-NEXT:   %1 = stablehlo.iota dim = 0 : tensor<1xi64>
// PLAIN-NEXT:   %c = stablehlo.constant dense<0> : tensor<1xi64>
// PLAIN-NEXT:   %2 = stablehlo.add %1, %c : tensor<1xi64>
// PLAIN-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<1xi64>
// PLAIN-NEXT:   %3 = stablehlo.multiply %2, %c_0 : tensor<1xi64>
// PLAIN-NEXT:   %4 = stablehlo.iota dim = 0 : tensor<1024xi64>
// PLAIN-NEXT:   %c_1 = stablehlo.constant dense<0> : tensor<1024xi64>
// PLAIN-NEXT:   %5 = stablehlo.add %4, %c_1 : tensor<1024xi64>
// PLAIN-NEXT:   %c_2 = stablehlo.constant dense<1> : tensor<1024xi64>
// PLAIN-NEXT:   %6 = stablehlo.multiply %5, %c_2 : tensor<1024xi64>
// PLAIN-NEXT:   %c_3 = stablehlo.constant dense<-1> : tensor<i64>
// PLAIN-NEXT:   %7 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<1024xi64>
// PLAIN-NEXT:   %8 = stablehlo.multiply %6, %7 : tensor<1024xi64>
// PLAIN-NEXT:   %9 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i64>) -> tensor<1024xi64>
// PLAIN-NEXT:   %10 = stablehlo.add %8, %9 : tensor<1024xi64>
// PLAIN-NEXT:   %c_4 = stablehlo.constant dense<-1> : tensor<i64>
// PLAIN-NEXT:   %11 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<1024xi64>
// PLAIN-NEXT:   %12 = stablehlo.add %10, %11 : tensor<1024xi64>
// PLAIN-NEXT:   %c_5 = stablehlo.constant dense<0> : tensor<1024xi64>
// PLAIN-NEXT:   %13 = stablehlo.compare GE, %12, %c_5 : (tensor<1024xi64>, tensor<1024xi64>) -> tensor<1024xi1>
// PLAIN-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PLAIN-NEXT:   %14 = stablehlo.pad %arg0, %cst, low = [0], high = [768], interior = [0] : (tensor<256xf64>, tensor<f64>) -> tensor<1024xf64>
// PLAIN-NEXT:   %15 = arith.mulf %14, %14 : tensor<1024xf64>
// PLAIN-NEXT:   %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// PLAIN-NEXT:   %c_7 = stablehlo.constant dense<768> : tensor<1xi64>
// PLAIN-NEXT:   %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// PLAIN-NEXT:   %c_9 = stablehlo.constant dense<-768> : tensor<1xi64>
// PLAIN-NEXT:   %c_10 = stablehlo.constant dense<0> : tensor<1xi64>
// PLAIN-NEXT:   %c_11 = stablehlo.constant dense<0> : tensor<i64>
// PLAIN-NEXT:   %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PLAIN-NEXT:   %16 = stablehlo.pad %arg1, %cst_12, low = [0], high = [768], interior = [0] : (tensor<256xf64>, tensor<f64>) -> tensor<1024xf64>
// PLAIN-NEXT:   %17 = stablehlo.select %13, %15, %16 : tensor<1024xi1>, tensor<1024xf64>
// PLAIN-NEXT:   %18 = stablehlo.dynamic_update_slice %16, %17, %c_11 : (tensor<1024xf64>, tensor<1024xf64>, tensor<i64>) -> tensor<1024xf64>
// PLAIN-NEXT:   %19 = stablehlo.slice %18 [0:256] : (tensor<1024xf64>) -> tensor<256xf64>
// PLAIN-NEXT:   return %arg0, %19, %arg2 : tensor<256xf64>, tensor<256xf64>, tensor<i32>
// PLAIN-NEXT: }
// PLAIN-NEXT: }
