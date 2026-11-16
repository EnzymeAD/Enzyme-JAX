// RUN: enzymexlamlir-opt --polygeist-mem2reg %s | FileCheck %s

// A capture the compiler filled with one wide integer, a copy of several of
// its members at once (an i320), and loaded whole for the members a kernel
// reads: each member is the bits of the integer at its place, shifted down
// and cut to its width.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f64 = dense<64> : vector<2xi64>>} {
  func.func @wide(%src: !llvm.ptr, %out: memref<3xi64>, %outp: memref<1x!llvm.ptr>) {
    %one = arith.constant 1 : i64
    %cap = llvm.alloca %one x !llvm.struct<(ptr, array<1 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    %w = llvm.load %src : !llvm.ptr -> i320
    llvm.store %w, %cap : i320, !llvm.ptr
    %v = "enzymexla.pointer2memref"(%cap) : (!llvm.ptr) -> memref<?x!llvm.struct<(ptr, array<1 x i64>, array<3 x i64>)>>
    %agg = affine.load %v[0] : memref<?x!llvm.struct<(ptr, array<1 x i64>, array<3 x i64>)>>
    %p = llvm.extractvalue %agg[0] : !llvm.struct<(ptr, array<1 x i64>, array<3 x i64>)>
    %n = llvm.extractvalue %agg[1, 0] : !llvm.struct<(ptr, array<1 x i64>, array<3 x i64>)>
    %s = llvm.extractvalue %agg[2, 1] : !llvm.struct<(ptr, array<1 x i64>, array<3 x i64>)>
    affine.store %n, %out[0] : memref<3xi64>
    affine.store %s, %out[1] : memref<3xi64>
    affine.store %p, %outp[0] : memref<1x!llvm.ptr>
    return
  }
}

// CHECK-LABEL: func.func @wide(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: !llvm.ptr, %[[V1:[^ ,:)]+]]: memref<3xi64>, %[[V2:[^ ,:)]+]]: memref<1x!llvm.ptr>) {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = arith.constant 1 : i64
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = llvm.alloca %[[V3]] x !llvm.struct<(ptr, array<1 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = llvm.load %[[V0]] : !llvm.ptr -> i320
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = arith.trunci %[[V5]] : i320 to i64
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = llvm.inttoptr %[[V6]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = arith.constant 64 : i320
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = arith.shrui %[[V5]], %[[V8]] : i320
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = arith.trunci %[[V9]] : i320 to i64
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = arith.constant 192 : i320
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = arith.shrui %[[V5]], %[[V11]] : i320
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = arith.trunci %[[V12]] : i320 to i64
// CHECK-NEXT: llvm.store %[[V5]], %[[V4]] : i320, !llvm.ptr
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V4]]) : (!llvm.ptr) -> memref<?x!llvm.struct<(ptr, array<1 x i64>, array<3 x i64>)>>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = affine.load %[[V14]][0] : memref<?x!llvm.struct<(ptr, array<1 x i64>, array<3 x i64>)>>
// CHECK-NEXT: affine.store %[[V10]], %[[V1]][0] : memref<3xi64>
// CHECK-NEXT: affine.store %[[V13]], %[[V1]][1] : memref<3xi64>
// CHECK-NEXT: affine.store %[[V7]], %[[V2]][0] : memref<1x!llvm.ptr>
// CHECK-NEXT: return
// CHECK-NEXT: }
