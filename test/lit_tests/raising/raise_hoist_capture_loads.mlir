// RUN: enzymexlamlir-opt %s --parallel-licm --canonicalize --raise-affine-to-stablehlo | FileCheck %s

// A kernel reads its lambda (the host's copy, an alloca made before the
// launch) for the pointer of the buffer it writes and a count. parallel-licm
// moves those reads out of the lanes and then ahead of the launch, so the
// raiser captures the buffer and takes the count as a scalar, rather than a
// tensor of pointers no form exists for.
module {
  func.func @capture(%c1i: i64) {
    %cap = llvm.alloca %c1i x !llvm.struct<(ptr, i32)> : (i64) -> !llvm.ptr
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c10, %c1, %c1) ({
      affine.parallel (%i) = (0) to (10) {
        %pv = "enzymexla.pointer2memref"(%cap) : (!llvm.ptr) -> memref<?x!llvm.ptr>
        %data = affine.load %pv[0] : memref<?x!llvm.ptr>
        %nv = "enzymexla.pointer2memref"(%cap) : (!llvm.ptr) -> memref<?xi32>
        %n = affine.load %nv[2] : memref<?xi32>
        %f = arith.sitofp %n : i32 to f64
        %dv = "enzymexla.pointer2memref"(%data) : (!llvm.ptr) -> memref<?xf64>
        affine.store %f, %dv[%i] : memref<?xf64>
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    func.return
  }
}

// CHECK-LABEL: func.func @capture(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: i64) {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = memref.alloca() : memref<f64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = arith.constant 10 : index
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = arith.constant 1 : index
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = llvm.alloca %[[V0]] x !llvm.struct<(ptr, i32)> : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V4]]) : (!llvm.ptr) -> memref<?x!llvm.ptr>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V4]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = affine.load %[[V5]][0] : memref<?x!llvm.ptr>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = affine.load %[[V6]][2] : memref<?xi32>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = arith.sitofp %[[V8]] : i32 to f64
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V7]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = gpu.alloc  () : memref<f64, 1>
// CHECK-NEXT: affine.store %[[V9]], %[[V1]][] : memref<f64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = arith.constant 8 : index
// CHECK-NEXT: enzymexla.memcpy  %[[V11]], %[[V1]], %[[V12]] : memref<f64, 1>, memref<f64>
// CHECK-NEXT: enzymexla.xla_wrapper @rxla$raised_0 (%[[V11]], %[[V10]]) : (memref<f64, 1>, memref<?xf64>) -> ()
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = arith.constant 0 : index
// CHECK-NEXT: gpu.dealloc  %[[V11]] : memref<f64, 1>
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK-LABEL: func.func private @rxla$raised_0(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<f64>, %[[V1:[^ ,:)]+]]: tensor<?xf64>) -> (tensor<f64>, tensor<?xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.add %[[V2]], %[[V3]] : tensor<10xi64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.multiply %[[V4]], %[[V5]] : tensor<10xi64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.get_dimension_size %[[V1]], dim = 0 : (tensor<?xf64>) -> tensor<i32>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.convert %[[V8]] : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.reshape %[[V9]] : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.constant dense<10> : tensor<1xi64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.add %[[V10]], %[[V7]] : tensor<1xi64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.subtract %[[V11]], %[[V12]] : tensor<1xi64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = stablehlo.maximum %[[V13]], %[[V14]] : tensor<1xi64>
// CHECK-NEXT: %[[V16:[^ ,:)]+]] = stablehlo.constant dense<-1> : tensor<1xi64>
// CHECK-NEXT: %[[V17:[^ ,:)]+]] = stablehlo.multiply %[[V7]], %[[V16]] : tensor<1xi64>
// CHECK-NEXT: %[[V18:[^ ,:)]+]] = stablehlo.multiply %[[V15]], %[[V16]] : tensor<1xi64>
// CHECK-NEXT: %[[V19:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V20:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V21:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [] : (tensor<f64>) -> tensor<10xf64>
// CHECK-NEXT: %[[V22:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V23:[^ ,:)]+]] = stablehlo.dynamic_pad %[[V1]], %[[V22]], %[[V7]], %[[V15]], %[[V19]] : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// CHECK-NEXT: %[[V24:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V23]], %[[V21]], %[[V20]] : (tensor<?xf64>, tensor<10xf64>, tensor<i64>) -> tensor<?xf64>
// CHECK-NEXT: %[[V25:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V26:[^ ,:)]+]] = stablehlo.dynamic_pad %[[V24]], %[[V25]], %[[V17]], %[[V18]], %[[V19]] : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// CHECK-NEXT: return %[[V0]], %[[V26]] : tensor<f64>, tensor<?xf64>
// CHECK-NEXT: }
