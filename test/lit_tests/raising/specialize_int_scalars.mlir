// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo="specialize_int_scalars=true" | FileCheck %s
// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s --check-prefix=DEFAULT

// y = flag ? y + a * x : y over 64 elements: the integer flag is specialized
// (a trailing input of the wrapper, no buffer of its own), the double factor
// is not.
func.func @flag_axpy(%y: memref<?xf64, 1>, %flag: i32, %a: f64, %x: memref<?xf64, 1>) {
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = "enzymexla.gpu_wrapper"(%c64, %c1, %c1, %c1, %c1, %c1) ({
    affine.parallel (%i) = (0) to (64) {
      %on = arith.cmpi ne, %flag, %c0_i32 : i32
      %xv = affine.load %x[%i] : memref<?xf64, 1>
      %yv = affine.load %y[%i] : memref<?xf64, 1>
      %p = arith.mulf %a, %xv : f64
      %s = arith.addf %yv, %p : f64
      %r = arith.select %on, %s, %yv : f64
      affine.store %r, %y[%i] : memref<?xf64, 1>
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK:  func.func @flag_axpy(%arg0: memref<?xf64, 1>, %arg1: i32, %arg2: f64, %arg3: memref<?xf64, 1>) {
// CHECK-NEXT:    %alloca = memref.alloca() : memref<f64>
// CHECK-NEXT:    %alloca_0 = memref.alloca() : memref<i32>
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c64 = arith.constant 64 : index
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    affine.store %arg1, %alloca_0[] : memref<i32>
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    %memref = gpu.alloc  () : memref<f64, 1>
// CHECK-NEXT:    affine.store %arg2, %alloca[] : memref<f64>
// CHECK-NEXT:    %c8 = arith.constant 8 : index
// CHECK-NEXT:    enzymexla.memcpy  %memref, %alloca, %c8 : memref<f64, 1>, memref<f64>
// CHECK-NEXT:    enzymexla.xla_wrapper @rxla$raised_0 (%arg3, %arg0, %memref, %arg1) <num_specialized = 1> : (memref<?xf64, 1>, memref<?xf64, 1>, memref<f64, 1>, i32) -> ()
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    gpu.dealloc  %memref : memref<f64, 1>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func private @rxla$raised_0(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<f64>, %arg3: tensor<i32>) -> (tensor<?xf64>, tensor<?xf64>, tensor<f64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<64xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<64xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c_0 : tensor<64xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<64xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_1 : tensor<64xi64>
// CHECK-NEXT:    %3 = arith.cmpi ne, %arg3, %c : tensor<i32>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %4 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf64>) -> tensor<i32>
// CHECK-NEXT:    %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:    %6 = stablehlo.reshape %5 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<64> : tensor<1xi64>
// CHECK-NEXT:    %7 = stablehlo.add %6, %c_2 : tensor<1xi64>
// CHECK-NEXT:    %8 = stablehlo.subtract %c_3, %7 : tensor<1xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %9 = stablehlo.maximum %8, %c_4 : tensor<1xi64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %10 = stablehlo.dynamic_pad %arg0, %cst, %c_2, %9, %c_5 : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// CHECK-NEXT:    %11 = stablehlo.slice %10 [0:64] : (tensor<?xf64>) -> tensor<64xf64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %12 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?xf64>) -> tensor<i32>
// CHECK-NEXT:    %13 = stablehlo.convert %12 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:    %14 = stablehlo.reshape %13 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<64> : tensor<1xi64>
// CHECK-NEXT:    %15 = stablehlo.add %14, %c_6 : tensor<1xi64>
// CHECK-NEXT:    %16 = stablehlo.subtract %c_7, %15 : tensor<1xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %17 = stablehlo.maximum %16, %c_8 : tensor<1xi64>
// CHECK-NEXT:    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %18 = stablehlo.dynamic_pad %arg1, %cst_9, %c_6, %17, %c_10 : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// CHECK-NEXT:    %19 = stablehlo.slice %18 [0:64] : (tensor<?xf64>) -> tensor<64xf64>
// CHECK-NEXT:    %20 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<64xf64>
// CHECK-NEXT:    %21 = arith.mulf %20, %11 : tensor<64xf64>
// CHECK-NEXT:    %22 = arith.addf %19, %21 : tensor<64xf64>
// CHECK-NEXT:    %23 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i1>) -> tensor<64xi1>
// CHECK-NEXT:    %24 = arith.select %23, %22, %19 : tensor<64xi1>, tensor<64xf64>
// CHECK-NEXT:    %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %25 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?xf64>) -> tensor<i32>
// CHECK-NEXT:    %26 = stablehlo.convert %25 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:    %27 = stablehlo.reshape %26 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:    %c_12 = stablehlo.constant dense<64> : tensor<1xi64>
// CHECK-NEXT:    %28 = stablehlo.add %27, %c_11 : tensor<1xi64>
// CHECK-NEXT:    %29 = stablehlo.subtract %c_12, %28 : tensor<1xi64>
// CHECK-NEXT:    %c_13 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %30 = stablehlo.maximum %29, %c_13 : tensor<1xi64>
// CHECK-NEXT:    %c_14 = stablehlo.constant dense<-1> : tensor<1xi64>
// CHECK-NEXT:    %31 = stablehlo.multiply %c_11, %c_14 : tensor<1xi64>
// CHECK-NEXT:    %32 = stablehlo.multiply %30, %c_14 : tensor<1xi64>
// CHECK-NEXT:    %c_15 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_16 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %33 = stablehlo.dynamic_pad %arg1, %cst_17, %c_11, %30, %c_15 : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// CHECK-NEXT:    %34 = stablehlo.dynamic_update_slice %33, %24, %c_16 : (tensor<?xf64>, tensor<64xf64>, tensor<i64>) -> tensor<?xf64>
// CHECK-NEXT:    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %35 = stablehlo.dynamic_pad %34, %cst_18, %31, %32, %c_15 : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// CHECK-NEXT:    return %arg0, %35, %arg2 : tensor<?xf64>, tensor<?xf64>, tensor<f64>
// CHECK-NEXT:  }

// DEFAULT:  func.func @flag_axpy(%arg0: memref<?xf64, 1>, %arg1: i32, %arg2: f64, %arg3: memref<?xf64, 1>) {
// DEFAULT-NEXT:    %alloca = memref.alloca() : memref<f64>
// DEFAULT-NEXT:    %alloca_0 = memref.alloca() : memref<i32>
// DEFAULT-NEXT:    %c1 = arith.constant 1 : index
// DEFAULT-NEXT:    %c64 = arith.constant 64 : index
// DEFAULT-NEXT:    %c0_i32 = arith.constant 0 : i32
// DEFAULT-NEXT:    %memref = gpu.alloc  () : memref<i32, 1>
// DEFAULT-NEXT:    affine.store %arg1, %alloca_0[] : memref<i32>
// DEFAULT-NEXT:    %c4 = arith.constant 4 : index
// DEFAULT-NEXT:    enzymexla.memcpy  %memref, %alloca_0, %c4 : memref<i32, 1>, memref<i32>
// DEFAULT-NEXT:    %memref_1 = gpu.alloc  () : memref<f64, 1>
// DEFAULT-NEXT:    affine.store %arg2, %alloca[] : memref<f64>
// DEFAULT-NEXT:    %c8 = arith.constant 8 : index
// DEFAULT-NEXT:    enzymexla.memcpy  %memref_1, %alloca, %c8 : memref<f64, 1>, memref<f64>
// DEFAULT-NEXT:    enzymexla.xla_wrapper @rxla$raised_0 (%memref, %arg3, %arg0, %memref_1) : (memref<i32, 1>, memref<?xf64, 1>, memref<?xf64, 1>, memref<f64, 1>) -> ()
// DEFAULT-NEXT:    %c0 = arith.constant 0 : index
// DEFAULT-NEXT:    gpu.dealloc  %memref_1 : memref<f64, 1>
// DEFAULT-NEXT:    gpu.dealloc  %memref : memref<i32, 1>
// DEFAULT-NEXT:    return
// DEFAULT-NEXT:  }
// DEFAULT:  func.func private @rxla$raised_0(%arg0: tensor<i32>, %arg1: tensor<?xf64>, %arg2: tensor<?xf64>, %arg3: tensor<f64>) -> (tensor<i32>, tensor<?xf64>, tensor<?xf64>, tensor<f64>) {
// DEFAULT-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// DEFAULT-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<64xi64>
// DEFAULT-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<64xi64>
// DEFAULT-NEXT:    %1 = stablehlo.add %0, %c_0 : tensor<64xi64>
// DEFAULT-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<64xi64>
// DEFAULT-NEXT:    %2 = stablehlo.multiply %1, %c_1 : tensor<64xi64>
// DEFAULT-NEXT:    %3 = arith.cmpi ne, %arg0, %c : tensor<i32>
// DEFAULT-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<1xi64>
// DEFAULT-NEXT:    %4 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?xf64>) -> tensor<i32>
// DEFAULT-NEXT:    %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<i64>
// DEFAULT-NEXT:    %6 = stablehlo.reshape %5 : (tensor<i64>) -> tensor<1xi64>
// DEFAULT-NEXT:    %c_3 = stablehlo.constant dense<64> : tensor<1xi64>
// DEFAULT-NEXT:    %7 = stablehlo.add %6, %c_2 : tensor<1xi64>
// DEFAULT-NEXT:    %8 = stablehlo.subtract %c_3, %7 : tensor<1xi64>
// DEFAULT-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<1xi64>
// DEFAULT-NEXT:    %9 = stablehlo.maximum %8, %c_4 : tensor<1xi64>
// DEFAULT-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// DEFAULT-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// DEFAULT-NEXT:    %10 = stablehlo.dynamic_pad %arg1, %cst, %c_2, %9, %c_5 : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// DEFAULT-NEXT:    %11 = stablehlo.slice %10 [0:64] : (tensor<?xf64>) -> tensor<64xf64>
// DEFAULT-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// DEFAULT-NEXT:    %12 = stablehlo.get_dimension_size %arg2, dim = 0 : (tensor<?xf64>) -> tensor<i32>
// DEFAULT-NEXT:    %13 = stablehlo.convert %12 : (tensor<i32>) -> tensor<i64>
// DEFAULT-NEXT:    %14 = stablehlo.reshape %13 : (tensor<i64>) -> tensor<1xi64>
// DEFAULT-NEXT:    %c_7 = stablehlo.constant dense<64> : tensor<1xi64>
// DEFAULT-NEXT:    %15 = stablehlo.add %14, %c_6 : tensor<1xi64>
// DEFAULT-NEXT:    %16 = stablehlo.subtract %c_7, %15 : tensor<1xi64>
// DEFAULT-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// DEFAULT-NEXT:    %17 = stablehlo.maximum %16, %c_8 : tensor<1xi64>
// DEFAULT-NEXT:    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// DEFAULT-NEXT:    %c_10 = stablehlo.constant dense<0> : tensor<1xi64>
// DEFAULT-NEXT:    %18 = stablehlo.dynamic_pad %arg2, %cst_9, %c_6, %17, %c_10 : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// DEFAULT-NEXT:    %19 = stablehlo.slice %18 [0:64] : (tensor<?xf64>) -> tensor<64xf64>
// DEFAULT-NEXT:    %20 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<f64>) -> tensor<64xf64>
// DEFAULT-NEXT:    %21 = arith.mulf %20, %11 : tensor<64xf64>
// DEFAULT-NEXT:    %22 = arith.addf %19, %21 : tensor<64xf64>
// DEFAULT-NEXT:    %23 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i1>) -> tensor<64xi1>
// DEFAULT-NEXT:    %24 = arith.select %23, %22, %19 : tensor<64xi1>, tensor<64xf64>
// DEFAULT-NEXT:    %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// DEFAULT-NEXT:    %25 = stablehlo.get_dimension_size %arg2, dim = 0 : (tensor<?xf64>) -> tensor<i32>
// DEFAULT-NEXT:    %26 = stablehlo.convert %25 : (tensor<i32>) -> tensor<i64>
// DEFAULT-NEXT:    %27 = stablehlo.reshape %26 : (tensor<i64>) -> tensor<1xi64>
// DEFAULT-NEXT:    %c_12 = stablehlo.constant dense<64> : tensor<1xi64>
// DEFAULT-NEXT:    %28 = stablehlo.add %27, %c_11 : tensor<1xi64>
// DEFAULT-NEXT:    %29 = stablehlo.subtract %c_12, %28 : tensor<1xi64>
// DEFAULT-NEXT:    %c_13 = stablehlo.constant dense<0> : tensor<1xi64>
// DEFAULT-NEXT:    %30 = stablehlo.maximum %29, %c_13 : tensor<1xi64>
// DEFAULT-NEXT:    %c_14 = stablehlo.constant dense<-1> : tensor<1xi64>
// DEFAULT-NEXT:    %31 = stablehlo.multiply %c_11, %c_14 : tensor<1xi64>
// DEFAULT-NEXT:    %32 = stablehlo.multiply %30, %c_14 : tensor<1xi64>
// DEFAULT-NEXT:    %c_15 = stablehlo.constant dense<0> : tensor<1xi64>
// DEFAULT-NEXT:    %c_16 = stablehlo.constant dense<0> : tensor<i64>
// DEFAULT-NEXT:    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// DEFAULT-NEXT:    %33 = stablehlo.dynamic_pad %arg2, %cst_17, %c_11, %30, %c_15 : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// DEFAULT-NEXT:    %34 = stablehlo.dynamic_update_slice %33, %24, %c_16 : (tensor<?xf64>, tensor<64xf64>, tensor<i64>) -> tensor<?xf64>
// DEFAULT-NEXT:    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// DEFAULT-NEXT:    %35 = stablehlo.dynamic_pad %34, %cst_18, %31, %32, %c_15 : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// DEFAULT-NEXT:    return %arg0, %arg1, %35, %arg3 : tensor<i32>, tensor<?xf64>, tensor<?xf64>, tensor<f64>
// DEFAULT-NEXT:  }
