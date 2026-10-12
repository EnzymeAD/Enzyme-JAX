// RUN: enzymexlamlir-opt --llvm-to-affine-access %s | FileCheck %s

// A by-value lambda a host function is handed by reference is loaded whole
// for the members its kernel reads: each member is loaded on its own, where
// the aggregate was, through a view of its type at its byte.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, f64 = dense<64> : vector<2xi64>>} {
  func.func @by_reference(%body: !llvm.ptr, %out: memref<2xi32>, %outp: memref<1x!llvm.ptr>) {
    %v = "enzymexla.pointer2memref"(%body) : (!llvm.ptr) -> memref<?x!llvm.struct<(i8, ptr, array<2 x struct<(i32, ptr, array<3 x i32>)>>)>>
    %agg = affine.load %v[0] : memref<?x!llvm.struct<(i8, ptr, array<2 x struct<(i32, ptr, array<3 x i32>)>>)>>
    %p = llvm.extractvalue %agg[1] : !llvm.struct<(i8, ptr, array<2 x struct<(i32, ptr, array<3 x i32>)>>)>
    %n = llvm.extractvalue %agg[2, 1, 2, 1] : !llvm.struct<(i8, ptr, array<2 x struct<(i32, ptr, array<3 x i32>)>>)>
    %m = llvm.extractvalue %agg[2, 0, 0] : !llvm.struct<(i8, ptr, array<2 x struct<(i32, ptr, array<3 x i32>)>>)>
    affine.store %n, %out[0] : memref<2xi32>
    affine.store %m, %out[1] : memref<2xi32>
    affine.store %p, %outp[0] : memref<1x!llvm.ptr>
    return
  }
}

// CHECK-LABEL: func.func @by_reference(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: !llvm.ptr, %[[V1:[^ ,:)]+]]: memref<2xi32>, %[[V2:[^ ,:)]+]]: memref<1x!llvm.ptr>) {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = llvm.getelementptr %[[V0]][16] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V3]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = affine.load %[[V4]][0] : memref<?xi32>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = llvm.getelementptr %[[V0]][68] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V6]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = affine.load %[[V7]][0] : memref<?xi32>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = llvm.getelementptr %[[V0]][8] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V9]]) : (!llvm.ptr) -> memref<?x!llvm.ptr>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = affine.load %[[V10]][0] : memref<?x!llvm.ptr>
// CHECK-NEXT: affine.store %[[V8]], %[[V1]][0] : memref<2xi32>
// CHECK-NEXT: affine.store %[[V5]], %[[V1]][1] : memref<2xi32>
// CHECK-NEXT: affine.store %[[V11]], %[[V2]][0] : memref<1x!llvm.ptr>
// CHECK-NEXT: return
// CHECK-NEXT: }
