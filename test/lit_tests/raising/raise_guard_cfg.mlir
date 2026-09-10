// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// The verify's error path keeps the host function in CFG form, so the guard
// is a cf.cond_br rather than an scf.if: the launch block is reached only
// through the `d < 25` edge, which bounds the axis at 24.
llvm.func @cfg_guard(%out: !llvm.ptr, %in: !llvm.ptr, %dbuf: !llvm.ptr) {
  %c1 = arith.constant 1 : index
  %c25 = arith.constant 25 : i32
  %dm = "enzymexla.pointer2memref"(%dbuf) : (!llvm.ptr) -> memref<?xi32>
  %d = affine.load %dm[0] : memref<?xi32>
  %ok = arith.cmpi slt, %d, %c25 : i32
  cf.cond_br %ok, ^launch, ^fail
^fail:
  llvm.unreachable
^launch:
  %di = arith.index_cast %d : i32 to index
  %om = "enzymexla.pointer2memref"(%out) : (!llvm.ptr) -> memref<?xf64>
  %im = "enzymexla.pointer2memref"(%in) : (!llvm.ptr) -> memref<?xf64>
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %di, %c1, %c1) ({
    affine.parallel (%e) = (0) to (1) {
      %scr = memref.alloca() : memref<32xf64>
      affine.parallel (%t) = (0) to (symbol(%di)) {
        %v = affine.load %im[%t] : memref<?xf64>
        affine.store %v, %scr[%t] : memref<32xf64>
        "enzymexla.barrier"(%t, %c1, %c1) : (index, index, index) -> ()
        %w = affine.load %scr[0] : memref<32xf64>
        affine.store %w, %om[%t] : memref<?xf64>
      }
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  llvm.return
}

// CHECK:    llvm.func @cfg_guard(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CHECK-NEXT:    %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c25_i32 = arith.constant 25 : i32
// CHECK-NEXT:    %0 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:    %1 = affine.load %0[0] : memref<?xi32>
// CHECK-NEXT:    %2 = arith.cmpi slt, %1, %c25_i32 : i32
// CHECK-NEXT:    cf.cond_br %2, ^bb2, ^bb1
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    llvm.unreachable
// CHECK-NEXT:  ^bb2:  // pred: ^bb0
// CHECK-NEXT:    %3 = arith.index_cast %1 : i32 to index
// CHECK-NEXT:    %4 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:    %5 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:    %memref = gpu.alloc  () : memref<i32, 1>
// CHECK-NEXT:    affine.store %1, %alloca[] : memref<i32>
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    enzymexla.memcpy  %memref, %alloca, %c4 : memref<i32, 1>, memref<i32>
// CHECK-NEXT:    enzymexla.xla_wrapper @rxla$raised_0 (%5, %4, %memref) : (memref<?xf64>, memref<?xf64>, memref<i32, 1>) -> ()
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    gpu.dealloc  %memref : memref<i32, 1>
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @rxla$raised_0(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<i32>) -> (tensor<?xf64>, tensor<?xf64>, tensor<i32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 0 : tensor<1xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %2 = stablehlo.add %1, %c_0 : tensor<1xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NEXT:    %3 = stablehlo.multiply %2, %c_1 : tensor<1xi64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x32xf64>
// CHECK-NEXT:    %4 = stablehlo.iota dim = 0 : tensor<24xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<24xi64>
// CHECK-NEXT:    %5 = stablehlo.add %4, %c_2 : tensor<24xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<1> : tensor<24xi64>
// CHECK-NEXT:    %6 = stablehlo.multiply %5, %c_3 : tensor<24xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %7 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<24xi64>
// CHECK-NEXT:    %8 = stablehlo.multiply %6, %7 : tensor<24xi64>
// CHECK-NEXT:    %9 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i64>) -> tensor<24xi64>
// CHECK-NEXT:    %10 = stablehlo.add %8, %9 : tensor<24xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %11 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i64>) -> tensor<24xi64>
// CHECK-NEXT:    %12 = stablehlo.add %10, %11 : tensor<24xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<24xi64>
// CHECK-NEXT:    %13 = stablehlo.compare GE, %12, %c_6 : (tensor<24xi64>, tensor<24xi64>) -> tensor<24xi1>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %14 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf64>) -> tensor<i32>
// CHECK-NEXT:    %15 = stablehlo.convert %14 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:    %16 = stablehlo.reshape %15 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<24> : tensor<1xi64>
// CHECK-NEXT:    %17 = stablehlo.add %16, %c_7 : tensor<1xi64>
// CHECK-NEXT:    %18 = stablehlo.subtract %c_8, %17 : tensor<1xi64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %19 = stablehlo.maximum %18, %c_9 : tensor<1xi64>
// CHECK-NEXT:    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %20 = stablehlo.dynamic_pad %arg0, %cst_10, %c_7, %19, %c_11 : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// CHECK-NEXT:    %21 = stablehlo.slice %20 [0:24] : (tensor<?xf64>) -> tensor<24xf64>
// CHECK-NEXT:    %c_12 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_13 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_14 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_15 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_16 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_17 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_18 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_19 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_20 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_21 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_22 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_23 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %22 = stablehlo.broadcast_in_dim %21, dims = [1] : (tensor<24xf64>) -> tensor<1x24xf64>
// CHECK-NEXT:    %23 = stablehlo.slice %cst [0:1, 0:24] : (tensor<1x32xf64>) -> tensor<1x24xf64>
// CHECK-NEXT:    %24 = stablehlo.broadcast_in_dim %13, dims = [0] : (tensor<24xi1>) -> tensor<24x1xi1>
// CHECK-NEXT:    %25 = stablehlo.broadcast_in_dim %22, dims = [1, 0] : (tensor<1x24xf64>) -> tensor<24x1xf64>
// CHECK-NEXT:    %26 = stablehlo.broadcast_in_dim %23, dims = [1, 0] : (tensor<1x24xf64>) -> tensor<24x1xf64>
// CHECK-NEXT:    %27 = stablehlo.select %24, %25, %26 : tensor<24x1xi1>, tensor<24x1xf64>
// CHECK-NEXT:    %28 = stablehlo.broadcast_in_dim %27, dims = [1, 0] : (tensor<24x1xf64>) -> tensor<1x24xf64>
// CHECK-NEXT:    %29 = stablehlo.dynamic_update_slice %cst, %28, %c_17, %c_23 : (tensor<1x32xf64>, tensor<1x24xf64>, tensor<i64>, tensor<i64>) -> tensor<1x32xf64>
// CHECK-NEXT:    %30 = stablehlo.slice %29 [0:1, 0:1] : (tensor<1x32xf64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %31 = stablehlo.reshape %30 : (tensor<1x1xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %c_24 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %32 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?xf64>) -> tensor<i32>
// CHECK-NEXT:    %33 = stablehlo.convert %32 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:    %34 = stablehlo.reshape %33 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:    %c_25 = stablehlo.constant dense<24> : tensor<1xi64>
// CHECK-NEXT:    %35 = stablehlo.add %34, %c_24 : tensor<1xi64>
// CHECK-NEXT:    %36 = stablehlo.subtract %c_25, %35 : tensor<1xi64>
// CHECK-NEXT:    %c_26 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %37 = stablehlo.maximum %36, %c_26 : tensor<1xi64>
// CHECK-NEXT:    %c_27 = stablehlo.constant dense<-1> : tensor<1xi64>
// CHECK-NEXT:    %38 = stablehlo.multiply %c_24, %c_27 : tensor<1xi64>
// CHECK-NEXT:    %39 = stablehlo.multiply %37, %c_27 : tensor<1xi64>
// CHECK-NEXT:    %c_28 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_29 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %40 = stablehlo.reshape %31 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:    %41 = stablehlo.broadcast_in_dim %40, dims = [] : (tensor<f64>) -> tensor<24xf64>
// CHECK-NEXT:    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %42 = stablehlo.dynamic_pad %arg1, %cst_30, %c_24, %37, %c_28 : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// CHECK-NEXT:    %43 = stablehlo.dynamic_slice %42, %c_29, sizes = [24] : (tensor<?xf64>, tensor<i64>) -> tensor<24xf64>
// CHECK-NEXT:    %44 = stablehlo.select %13, %41, %43 : tensor<24xi1>, tensor<24xf64>
// CHECK-NEXT:    %45 = stablehlo.dynamic_update_slice %42, %44, %c_29 : (tensor<?xf64>, tensor<24xf64>, tensor<i64>) -> tensor<?xf64>
// CHECK-NEXT:    %cst_31 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %46 = stablehlo.dynamic_pad %45, %cst_31, %38, %39, %c_28 : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// CHECK-NEXT:    return %arg0, %46, %arg2 : tensor<?xf64>, tensor<?xf64>, tensor<i32>
// CHECK-NEXT:  }
