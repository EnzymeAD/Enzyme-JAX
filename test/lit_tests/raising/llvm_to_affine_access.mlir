// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" | FileCheck %s

// CHECK-LABEL:   func.func @test_load_store_conversion(
// CHECK-SAME:                                          %[[VAL_0:[^:]*]]: !llvm.ptr<1>,
// CHECK-SAME:                                          %[[VAL_1:[^:]*]]: i64) {
// CHECK:           %[[VAL_2:.*]] = "enzymexla.pointer2memref"(%[[VAL_0]]) : (!llvm.ptr<1>) -> memref<?xi64, 1>
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK:           %[[VAL_5:.*]] = affine.load %[[VAL_2]][symbol(%[[VAL_3]])] : memref<?xi64, 1>
// CHECK:           %[[VAL_6:.*]] = llvm.mul %[[VAL_5]], %[[VAL_5]] : i64
// CHECK:           affine.store %[[VAL_6]], %[[VAL_2]][symbol(%[[VAL_4]])] : memref<?xi64, 1>
// CHECK:           return
// CHECK:         }
func.func @test_load_store_conversion(%arg0: !llvm.ptr<1>, %idx: i64) {
  %0 = llvm.getelementptr inbounds %arg0[%idx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
  %1 = llvm.load %0 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
  %2 = llvm.mul %1, %1 : i64

  llvm.store %2, %0 {alignment = 1 : i64} : i64, !llvm.ptr<1>

  return
}


// -----

// CHECK-LABEL:   func.func @test_multidim_load_store(
// CHECK-SAME:                                        %[[VAL_0:[^:]*]]: !llvm.ptr<1>,
// CHECK-SAME:                                        %[[VAL_1:[^:]*]]: i64,
// CHECK-SAME:                                        %[[VAL_2:[^:]*]]: i64) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_4:.*]] = "enzymexla.pointer2memref"(%[[VAL_0]]) : (!llvm.ptr<1>) -> memref<?xi64, 1>
// CHECK:           %[[VAL_5:.*]] = arith.index_cast %[[VAL_2]] : i64 to index
// CHECK:           %[[VAL_6:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK:           %[[VAL_7:.*]] = affine.load %[[VAL_4]][symbol(%[[VAL_6]]) * 8 + symbol(%[[VAL_5]])] : memref<?xi64, 1>
// CHECK:           %[[VAL_8:.*]] = llvm.add %[[VAL_1]], %[[VAL_3]] : i64
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[VAL_8]] : i64 to index
// CHECK:           %[[VAL_10:.*]] = llvm.add %[[VAL_2]], %[[VAL_3]] : i64
// CHECK:           %[[VAL_11:.*]] = arith.index_cast %[[VAL_10]] : i64 to index
// CHECK:           affine.store %[[VAL_7]], %[[VAL_4]][symbol(%[[VAL_9]]) * 8 + symbol(%[[VAL_11]])] : memref<?xi64, 1>
// CHECK:           return
// CHECK:         }
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


// -----

// CHECK-LABEL:   func.func @test_struct_access(
// CHECK-SAME:                                  %[[VAL_0:[^:]*]]: !llvm.ptr) {
// CHECK:           %[[VAL_1:.*]] = "enzymexla.pointer2memref"(%[[VAL_0]]) : (!llvm.ptr) -> memref<?xi64>
// CHECK:           %[[VAL_2:.*]] = affine.load %[[VAL_1]][0] : memref<?xi64>
// CHECK:           affine.store %[[VAL_2]], %[[VAL_1]][0] : memref<?xi64>
// CHECK:           return
// CHECK:         }
func.func @test_struct_access(%arg0: !llvm.ptr) {
  %ptr = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64)>
  %val = llvm.load %ptr : !llvm.ptr -> i64

  llvm.store %val, %ptr : i64, !llvm.ptr

  return
}


// CHEC-K: %[[MEMREF:.*]] = enzymexla.at_addr %arg0 : !llvm.ptr to memref<?xf32>
// CHEC-K: %[[LOAD:.*]] = memref.load %[[MEMREF]][%c0] : memref<?xf32>
// CHEC-K: memref.store %[[VAL]], %[[MEMREF]][%c0] : memref<?xf32>
