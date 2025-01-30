// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" | FileCheck %s

func.func @test_load_store_conversion(%arg0: !llvm.ptr<1>, %idx: i64) {
  %0 = llvm.getelementptr inbounds %arg0[%idx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
  %1 = llvm.load %0 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
  %2 = llvm.mul %1, %1 : i64

  llvm.store %2, %0 {alignment = 1 : i64} : i64, !llvm.ptr<1>

  return
}

// CHECK-LABEL: func @test_load_store_conversion
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<1>
// CHECK-SAME: %[[ARG1:.*]]: i64
// CHECK: %[[MEMREF:.*]] = "enzymexla.pointer2memref"(%[[ARG0]]) {{.*}} memref<?xi8, 1>
// CHECK: %[[IDX:.*]] = arith.index_cast %[[ARG1]]
// CHECK: affine.vector_load %[[MEMREF]][symbol(%[[IDX]]) * 8] {{.*}} vector<8xi8>
// CHECK: affine.vector_store

// -----

func.func @test_multidim_load_store(%arg0: !llvm.ptr<1>, %idx1: i64, %idx2: i64) {
  %c1 = llvm.mlir.constant(1 : index) : i64
  %ptr = llvm.getelementptr %arg0[%idx1, %idx2] : (!llvm.ptr<1>, i64, i64) -> !llvm.ptr<1>, !llvm.array<8 x i64>
  %val = llvm.load %ptr : !llvm.ptr<1> -> i64

  %idx1p1 = llvm.add %idx1, %c1 : i64
  %idx2p1 = llvm.add %idx2, %c1 : i64

  %ptr_str = llvm.getelementptr %arg0[%idx1p1, %idx2p1] : (!llvm.ptr<1>, i64, i64) -> !llvm.ptr<1>, !llvm.array<8 x i64>
  llvm.store %val, %ptr_str : i64, !llvm.ptr<1>

  return
}

// CHECK-LABEL: func @test_multidim_load_store
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<1>,
// CHECK-SAME: %[[ARG1:.*]]: i64,
// CHECK-SAME: %[[ARG2:.*]]: i64
// CHECK: %[[MEMREF:.*]] = "enzymexla.pointer2memref"(%[[ARG0]]) {{.*}} memref<?xi8, 1>
// CHECK-DAG: %[[IDX1:.*]] = arith.index_cast %[[ARG1]]
// CHECK-DAG: %[[IDX2:.*]] = arith.index_cast %[[ARG2]]
// CHECK: affine.vector_load %[[MEMREF]][symbol(%[[IDX1]]) * 64 + symbol(%[[IDX2]]) * 8] {{.*}} vector<8xi8>
// CHECK: affine.vector_store

// -----

func.func @test_struct_access(%arg0: !llvm.ptr) {
  %ptr = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64)>
  %val = llvm.load %ptr : !llvm.ptr -> i64

  llvm.store %val, %ptr : i64, !llvm.ptr

  return
}

// CHECK-LABEL: func @test_struct_access
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr
// CHECK: %[[MEMREF:.*]] = "enzymexla.pointer2memref"(%[[ARG0]]) {{.*}} memref<?xi8>
// CHECK: affine.vector_load %[[MEMREF]][0] {{.*}} vector<8xi8>
// CHECK: affine.vector_store

// CHEC-K: %[[MEMREF:.*]] = enzymexla.at_addr %arg0 : !llvm.ptr to memref<?xf32>
// CHEC-K: %[[LOAD:.*]] = memref.load %[[MEMREF]][%c0] : memref<?xf32>
// CHEC-K: memref.store %[[VAL]], %[[MEMREF]][%c0] : memref<?xf32>
