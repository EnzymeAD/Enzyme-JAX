// RUN: enzymexlamlir-opt %s --parallel-licm | FileCheck %s

// A kernel reads its by-value lambda (the host's byval copy) for the pointer
// of the buffer it writes and a count: the reads leave the lanes, which run,
// and then the launch, ahead of it on the host.
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

// The kernel writes the copy before reading it: the read stays in it.
func.func @written(%cap: !llvm.ptr {llvm.byval = !llvm.struct<(ptr, i32)>, llvm.align = 8 : i64}, %v: i32) {
  %c1 = arith.constant 1 : index
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
    %nv = "enzymexla.pointer2memref"(%cap) : (!llvm.ptr) -> memref<?xi32>
    affine.store %v, %nv[2] : memref<?xi32>
    %n = affine.load %nv[2] : memref<?xi32>
    %dv = "enzymexla.pointer2memref"(%cap) : (!llvm.ptr) -> memref<?xi32>
    affine.store %n, %dv[3] : memref<?xi32>
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  func.return
}

// A read through a pointer that is not a host object's (a device buffer may
// be passed this way) stays in the kernel.
func.func @device(%buf: !llvm.ptr, %out: memref<?xf64>) {
  %c1 = arith.constant 1 : index
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
    %bv = "enzymexla.pointer2memref"(%buf) : (!llvm.ptr) -> memref<?xf64>
    %x = affine.load %bv[0] : memref<?xf64>
    affine.store %x, %out[0] : memref<?xf64>
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  func.return
}

// CHECK-LABEL: func.func @capture(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: !llvm.ptr {llvm.align = 8 : i64, llvm.byval = !llvm.struct<(ptr, i32)>}) {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = arith.constant 1 : index
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = arith.constant 10 : index
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V0]]) : (!llvm.ptr) -> memref<?x!llvm.ptr>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V0]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = affine.load %[[V3]][0] : memref<?x!llvm.ptr>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = affine.load %[[V4]][2] : memref<?xi32>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = arith.sitofp %[[V6]] : i32 to f64
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V5]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = "enzymexla.gpu_wrapper"(%[[V1]], %[[V1]], %[[V1]], %[[V2]], %[[V1]], %[[V1]]) ({
// CHECK-NEXT: affine.parallel (%[[V10:[^ ,:)]+]]) = (0) to (10) {
// CHECK-NEXT: affine.store %[[V7]], %[[V8]][%[[V10]]] : memref<?xf64>
// CHECK-NEXT: }
// CHECK-NEXT: "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT: }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @written(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: !llvm.ptr {llvm.align = 8 : i64, llvm.byval = !llvm.struct<(ptr, i32)>}, %[[V1:[^ ,:)]+]]: i32) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = arith.constant 1 : index
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V0]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V0]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = "enzymexla.gpu_wrapper"(%[[V2]], %[[V2]], %[[V2]], %[[V2]], %[[V2]], %[[V2]]) ({
// CHECK-NEXT: affine.store %[[V1]], %[[V3]][2] : memref<?xi32>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = affine.load %[[V3]][2] : memref<?xi32>
// CHECK-NEXT: affine.store %[[V6]], %[[V4]][3] : memref<?xi32>
// CHECK-NEXT: "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT: }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @device(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: !llvm.ptr, %[[V1:[^ ,:)]+]]: memref<?xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = arith.constant 1 : index
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V0]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = "enzymexla.gpu_wrapper"(%[[V2]], %[[V2]], %[[V2]], %[[V2]], %[[V2]], %[[V2]]) ({
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = affine.load %[[V3]][0] : memref<?xf64>
// CHECK-NEXT: affine.store %[[V5]], %[[V1]][0] : memref<?xf64>
// CHECK-NEXT: "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT: }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT: return
// CHECK-NEXT: }
