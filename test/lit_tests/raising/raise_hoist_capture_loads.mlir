// RUN: enzymexlamlir-opt --raise-affine-to-stablehlo %s | FileCheck %s

// A kernel reads its by-value lambda (here the host's byval copy) for the
// pointer of the buffer it writes and a count: those reads are the host's to
// make before the launch, so the buffer is captured and the count is a
// scalar, rather than a tensor of pointers no form exists for.
module {
  func.func @capture(%cap: !llvm.ptr {llvm.byval = !llvm.struct<(ptr, i32)>, llvm.align = 8 : i64}) {
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
// CHECK-SAME: %[[V0:[^ ,:)]+]]: !llvm.ptr {llvm.align = 8 : i64, llvm.byval = !llvm.struct<(ptr, i32)>}) {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = memref.alloca() : memref<i32>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = arith.constant 1 : index
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = arith.constant 10 : index
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V0]]) : (!llvm.ptr) -> memref<?x!llvm.ptr>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = affine.load %[[V4]][0] : memref<?x!llvm.ptr>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V0]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = affine.load %[[V6]][2] : memref<?xi32>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = gpu.alloc  () : memref<i32, 1>
// CHECK-NEXT: affine.store %[[V7]], %[[V1]][] : memref<i32>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = arith.constant 4 : index
// CHECK-NEXT: enzymexla.memcpy  %[[V8]], %[[V1]], %[[V9]] : memref<i32, 1>, memref<i32>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V5]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT: enzymexla.xla_wrapper @rxla$raised_0 (%[[V8]], %[[V10]]) : (memref<i32, 1>, memref<?xf64>) -> ()
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = arith.constant 0 : index
// CHECK-NEXT: gpu.dealloc  %[[V8]] : memref<i32, 1>
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @rxla$raised_0(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<i32>, %[[V1:[^ ,:)]+]]: tensor<?xf64>) -> (tensor<i32>, tensor<?xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.add %[[V2]], %[[V3]] : tensor<10xi64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.multiply %[[V4]], %[[V5]] : tensor<10xi64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = arith.sitofp %[[V0]] : tensor<i32> to tensor<f64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.get_dimension_size %[[V1]], dim = 0 : (tensor<?xf64>) -> tensor<i32>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.convert %[[V9]] : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.reshape %[[V10]] : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.constant dense<10> : tensor<1xi64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.add %[[V11]], %[[V8]] : tensor<1xi64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.subtract %[[V12]], %[[V13]] : tensor<1xi64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V16:[^ ,:)]+]] = stablehlo.maximum %[[V14]], %[[V15]] : tensor<1xi64>
// CHECK-NEXT: %[[V17:[^ ,:)]+]] = stablehlo.constant dense<-1> : tensor<1xi64>
// CHECK-NEXT: %[[V18:[^ ,:)]+]] = stablehlo.multiply %[[V8]], %[[V17]] : tensor<1xi64>
// CHECK-NEXT: %[[V19:[^ ,:)]+]] = stablehlo.multiply %[[V16]], %[[V17]] : tensor<1xi64>
// CHECK-NEXT: %[[V20:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V21:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V22:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V7]], dims = [] : (tensor<f64>) -> tensor<10xf64>
// CHECK-NEXT: %[[V23:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V24:[^ ,:)]+]] = stablehlo.dynamic_pad %[[V1]], %[[V23]], %[[V8]], %[[V16]], %[[V20]] : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// CHECK-NEXT: %[[V25:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V24]], %[[V22]], %[[V21]] : (tensor<?xf64>, tensor<10xf64>, tensor<i64>) -> tensor<?xf64>
// CHECK-NEXT: %[[V26:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V27:[^ ,:)]+]] = stablehlo.dynamic_pad %[[V25]], %[[V26]], %[[V18]], %[[V19]], %[[V20]] : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// CHECK-NEXT: return %[[V0]], %[[V27]] : tensor<i32>, tensor<?xf64>
// CHECK-NEXT: }
