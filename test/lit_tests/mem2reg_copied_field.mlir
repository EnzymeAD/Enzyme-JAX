// RUN: enzymexlamlir-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

// A capture built member by member, one member copied in from memory, and
// loaded whole for the members a kernel reads: each member taken out of the
// load is a slot of its own, answered by the store or the copy that filled it.
// The copy reaches over the member and more, and leaves in it the bytes of its
// source at the member's place in the copy.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f64 = dense<64> : vector<2xi64>>} {
  func.func @members(%mat: !llvm.ptr, %a: f64, %out: memref<2xf64>) {
    %one = arith.constant 1 : i64
    %c16 = arith.constant 16 : i64
    %cap = llvm.alloca %one x !llvm.struct<(array<2 x f64>, struct<(f64, f64)>)> : (i64) -> !llvm.ptr
    %f = "enzymexla.pointer2memref"(%cap) : (!llvm.ptr) -> memref<?xf64>
    affine.store %a, %f[0] : memref<?xf64>
    affine.store %a, %f[1] : memref<?xf64>
    %m = llvm.getelementptr %cap[16] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.memcpy"(%m, %mat, %c16) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %v = "enzymexla.pointer2memref"(%cap) : (!llvm.ptr) -> memref<?x!llvm.struct<(array<2 x f64>, struct<(f64, f64)>)>>
    %agg = affine.load %v[0] : memref<?x!llvm.struct<(array<2 x f64>, struct<(f64, f64)>)>>
    %y = llvm.extractvalue %agg[1, 1] : !llvm.struct<(array<2 x f64>, struct<(f64, f64)>)>
    %z = llvm.extractvalue %agg[0, 1] : !llvm.struct<(array<2 x f64>, struct<(f64, f64)>)>
    affine.store %y, %out[1] : memref<2xf64>
    affine.store %z, %out[0] : memref<2xf64>
    return
  }
}

// CHECK-LABEL: func.func @members(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: !llvm.ptr, %[[V1:[^ ,:)]+]]: f64, %[[V2:[^ ,:)]+]]: memref<2xf64>) {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = arith.constant 1 : i64
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = arith.constant 16 : i64
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = llvm.getelementptr %[[V0]][8] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = llvm.load %[[V5]] <alignment = 1> : !llvm.ptr -> f64
// CHECK-NEXT: affine.store %[[V6]], %[[V2]][1] : memref<2xf64>
// CHECK-NEXT: affine.store %[[V1]], %[[V2]][0] : memref<2xf64>
// CHECK-NEXT: return
// CHECK-NEXT: }

// -----

// The same while something reads the capture at an index of unknown range,
// so the capture stays: the member still forwards.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f64 = dense<64> : vector<2xi64>>} {
  func.func @read_anywhere(%mat: !llvm.ptr, %a: f64, %i: index, %out: memref<2xf64>) {
    %one = arith.constant 1 : i64
    %c16 = arith.constant 16 : i64
    %cap = llvm.alloca %one x !llvm.struct<(array<2 x f64>, struct<(f64, f64)>)> : (i64) -> !llvm.ptr
    %f = "enzymexla.pointer2memref"(%cap) : (!llvm.ptr) -> memref<?xf64>
    affine.store %a, %f[0] : memref<?xf64>
    affine.store %a, %f[1] : memref<?xf64>
    %m = llvm.getelementptr %cap[16] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.memcpy"(%m, %mat, %c16) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %x = memref.load %f[%i] : memref<?xf64>
    affine.store %x, %out[0] : memref<2xf64>
    %v = "enzymexla.pointer2memref"(%cap) : (!llvm.ptr) -> memref<?x!llvm.struct<(array<2 x f64>, struct<(f64, f64)>)>>
    %agg = affine.load %v[0] : memref<?x!llvm.struct<(array<2 x f64>, struct<(f64, f64)>)>>
    %y = llvm.extractvalue %agg[1, 1] : !llvm.struct<(array<2 x f64>, struct<(f64, f64)>)>
    affine.store %y, %out[1] : memref<2xf64>
    return
  }
}

// CHECK-LABEL: func.func @read_anywhere(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: !llvm.ptr, %[[V1:[^ ,:)]+]]: f64, %[[V2:[^ ,:)]+]]: index, %[[V3:[^ ,:)]+]]: memref<2xf64>) {
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = arith.constant 1 : i64
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = arith.constant 16 : i64
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = llvm.alloca %[[V4]] x !llvm.struct<(array<2 x f64>, struct<(f64, f64)>)> : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V6]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT: affine.store %[[V1]], %[[V7]][0] : memref<?xf64>
// CHECK-NEXT: affine.store %[[V1]], %[[V7]][1] : memref<?xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = llvm.getelementptr %[[V6]][16] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = llvm.getelementptr %[[V0]][8] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = llvm.load %[[V9]] <alignment = 1> : !llvm.ptr -> f64
// CHECK-NEXT: "llvm.intr.memcpy"(%[[V8]], %[[V0]], %[[V5]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = memref.load %[[V7]][%[[V2]]] : memref<?xf64>
// CHECK-NEXT: affine.store %[[V11]], %[[V3]][0] : memref<2xf64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V6]]) : (!llvm.ptr) -> memref<?x!llvm.struct<(array<2 x f64>, struct<(f64, f64)>)>>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = affine.load %[[V12]][0] : memref<?x!llvm.struct<(array<2 x f64>, struct<(f64, f64)>)>>
// CHECK-NEXT: affine.store %[[V10]], %[[V3]][1] : memref<2xf64>
// CHECK-NEXT: return
// CHECK-NEXT: }
