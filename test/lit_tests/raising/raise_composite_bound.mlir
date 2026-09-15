// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// The lane extent is a composite of a guarded value (3*D with D < 9): the
// bound flows through the expression and the guard compares the composite.
func.func @composite(%out: memref<32xf64, 1>, %in: memref<32xf64, 1>, %dbuf: memref<i32, 1>, %unused: index) {
  %c1 = arith.constant 1 : index
  %c9 = arith.constant 9 : i32
  %d = affine.load %dbuf[] : memref<i32, 1>
  %ok = arith.cmpi slt, %d, %c9 : i32
  scf.if %ok {
    %di = arith.index_cast %d : i32 to index
    %ext = affine.apply affine_map<()[s0] -> (s0 * 3)>()[%di]
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %ext, %c1, %c1) ({
      affine.parallel (%e) = (0) to (1) {
        %scr = memref.alloca() : memref<32xf64>
        affine.parallel (%t) = (0) to (symbol(%di) * 3) {
          %v = affine.load %in[%t] : memref<32xf64, 1>
          affine.store %v, %scr[%t] : memref<32xf64>
          "enzymexla.barrier"(%t, %c1, %c1) : (index, index, index) -> ()
          %w = affine.load %scr[0] : memref<32xf64>
          affine.store %w, %out[%t] : memref<32xf64, 1>
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
  }
  return
}

// CHECK:  #map = affine_map<()[s0] -> (s0 * 3)>
// CHECK-NEXT:module {
// CHECK-NEXT:  func.func @composite(%arg0: memref<32xf64, 1>, %arg1: memref<32xf64, 1>, %arg2: memref<i32, 1>, %arg3: index) {
// CHECK-NEXT:    %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c9_i32 = arith.constant 9 : i32
// CHECK-NEXT:    %0 = affine.load %arg2[] : memref<i32, 1>
// CHECK-NEXT:    %1 = arith.cmpi slt, %0, %c9_i32 : i32
// CHECK-NEXT:    scf.if %1 {
// CHECK-NEXT:      %2 = arith.index_cast %0 : i32 to index
// CHECK-NEXT:      %3 = affine.apply #map()[%2]
// CHECK-NEXT:      %memref = gpu.alloc  () : memref<i32, 1>
// CHECK-NEXT:      affine.store %0, %alloca[] : memref<i32>
// CHECK-NEXT:      %c4 = arith.constant 4 : index
// CHECK-NEXT:      enzymexla.memcpy  %memref, %alloca, %c4 : memref<i32, 1>, memref<i32>
// CHECK-NEXT:      enzymexla.xla_wrapper @rxla$raised_0 (%memref, %arg1, %arg0) : (memref<i32, 1>, memref<32xf64, 1>, memref<32xf64, 1>) -> ()
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      gpu.dealloc  %memref : memref<i32, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @rxla$raised_0(%arg0: tensor<i32>, %arg1: tensor<32xf64>, %arg2: tensor<32xf64>) -> (tensor<i32>, tensor<32xf64>, tensor<32xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.convert %arg0 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 0 : tensor<1xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %2 = stablehlo.add %1, %c_0 : tensor<1xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NEXT:    %3 = stablehlo.multiply %2, %c_1 : tensor<1xi64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x32xf64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %4 = stablehlo.multiply %0, %c_2 : tensor<i64>
// CHECK-NEXT:    %5 = stablehlo.iota dim = 0 : tensor<24xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<24xi64>
// CHECK-NEXT:    %6 = stablehlo.add %5, %c_3 : tensor<24xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<1> : tensor<24xi64>
// CHECK-NEXT:    %7 = stablehlo.multiply %6, %c_4 : tensor<24xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i64>) -> tensor<24xi64>
// CHECK-NEXT:    %9 = stablehlo.multiply %7, %8 : tensor<24xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %10 = stablehlo.multiply %0, %c_6 : tensor<i64>
// CHECK-NEXT:    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i64>) -> tensor<24xi64>
// CHECK-NEXT:    %12 = stablehlo.add %9, %11 : tensor<24xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %13 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i64>) -> tensor<24xi64>
// CHECK-NEXT:    %14 = stablehlo.add %12, %13 : tensor<24xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<24xi64>
// CHECK-NEXT:    %15 = stablehlo.compare GE, %14, %c_8 : (tensor<24xi64>, tensor<24xi64>) -> tensor<24xi1>
// CHECK-NEXT:    %16 = stablehlo.slice %arg1 [0:24] : (tensor<32xf64>) -> tensor<24xf64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_12 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_13 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_14 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_15 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_16 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_17 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_18 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_19 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_20 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %17 = stablehlo.broadcast_in_dim %16, dims = [1] : (tensor<24xf64>) -> tensor<1x24xf64>
// CHECK-NEXT:    %18 = stablehlo.slice %cst [0:1, 0:24] : (tensor<1x32xf64>) -> tensor<1x24xf64>
// CHECK-NEXT:    %19 = stablehlo.broadcast_in_dim %15, dims = [0] : (tensor<24xi1>) -> tensor<24x1xi1>
// CHECK-NEXT:    %20 = stablehlo.broadcast_in_dim %17, dims = [1, 0] : (tensor<1x24xf64>) -> tensor<24x1xf64>
// CHECK-NEXT:    %21 = stablehlo.broadcast_in_dim %18, dims = [1, 0] : (tensor<1x24xf64>) -> tensor<24x1xf64>
// CHECK-NEXT:    %22 = stablehlo.select %19, %20, %21 : tensor<24x1xi1>, tensor<24x1xf64>
// CHECK-NEXT:    %23 = stablehlo.broadcast_in_dim %22, dims = [1, 0] : (tensor<24x1xf64>) -> tensor<1x24xf64>
// CHECK-NEXT:    %24 = stablehlo.dynamic_update_slice %cst, %23, %c_14, %c_20 : (tensor<1x32xf64>, tensor<1x24xf64>, tensor<i64>, tensor<i64>) -> tensor<1x32xf64>
// CHECK-NEXT:    %25 = stablehlo.slice %24 [0:1, 0:1] : (tensor<1x32xf64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %26 = stablehlo.reshape %25 : (tensor<1x1xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %c_21 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_22 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_23 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_24 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_25 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_26 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %27 = stablehlo.reshape %26 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:    %28 = stablehlo.broadcast_in_dim %27, dims = [] : (tensor<f64>) -> tensor<24xf64>
// CHECK-NEXT:    %29 = stablehlo.slice %arg2 [0:24] : (tensor<32xf64>) -> tensor<24xf64>
// CHECK-NEXT:    %30 = stablehlo.select %15, %28, %29 : tensor<24xi1>, tensor<24xf64>
// CHECK-NEXT:    %31 = stablehlo.dynamic_update_slice %arg2, %30, %c_26 : (tensor<32xf64>, tensor<24xf64>, tensor<i64>) -> tensor<32xf64>
// CHECK-NEXT:    return %arg0, %arg1, %31 : tensor<i32>, tensor<32xf64>, tensor<32xf64>
// CHECK-NEXT:  }
