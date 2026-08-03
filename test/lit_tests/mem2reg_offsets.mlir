// RUN: enzymexlamlir-opt %s -polygeist-mem2reg -split-input-file | FileCheck %s

// An access at index zero of a view names the start of the allocation, which is
// where an index-free access lands, whatever the rank the view was given.
llvm.func @zero_index_meets_llvm_load(%val: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.array<4 x array<4 x f32>> : (i32) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%mem) : (!llvm.ptr) -> memref<?x?xf32>
  affine.store %val, %view[0, 0] : memref<?x?xf32>
  %loaded = llvm.load %mem : !llvm.ptr -> f32
  llvm.return %loaded : f32
}

// CHECK-LABEL: llvm.func @zero_index_meets_llvm_load(
// CHECK-SAME: %[[VAL:[a-z0-9]+]]: f32
// CHECK-NOT: llvm.load
// CHECK: llvm.return %[[VAL]] : f32

// -----

// The same, the other way around: the value written without indices is read
// back through a zero index of a view.
llvm.func @llvm_store_meets_zero_index(%val: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.array<4 x array<4 x f32>> : (i32) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%mem) : (!llvm.ptr) -> memref<?x?xf32>
  llvm.store %val, %mem : f32, !llvm.ptr
  %loaded = memref.load %view[%c0, %c0] : memref<?x?xf32>
  llvm.return %loaded : f32
}

// CHECK-LABEL: llvm.func @llvm_store_meets_zero_index(
// CHECK-SAME: %[[VAL:[a-z0-9]+]]: f32
// CHECK-NOT: memref.load
// CHECK: llvm.return %[[VAL]] : f32

// -----

// A field written and read through the same offset of the allocation is one
// slot, which the offsets say without either access being index-free.
llvm.func @gep_same_field(%val: f32) -> f32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.struct<(i64, f32, f32)> : (i32) -> !llvm.ptr
  %f1 = llvm.getelementptr %mem[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, f32, f32)>
  llvm.store %val, %f1 : f32, !llvm.ptr
  %f1again = llvm.getelementptr %mem[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, f32, f32)>
  %loaded = llvm.load %f1again : !llvm.ptr -> f32
  llvm.return %loaded : f32
}

// CHECK-LABEL: llvm.func @gep_same_field(
// CHECK-SAME: %[[VAL:[a-z0-9]+]]: f32
// CHECK-NOT: llvm.load
// CHECK: llvm.return %[[VAL]] : f32

// -----

// A store to a different field of the same allocation is a different slot, so
// it neither forwards to the load nor stops the store that does.
llvm.func @gep_other_field(%val: f32, %other: f32) -> f32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.struct<(i64, f32, f32)> : (i32) -> !llvm.ptr
  %f1 = llvm.getelementptr %mem[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, f32, f32)>
  llvm.store %val, %f1 : f32, !llvm.ptr
  %f2 = llvm.getelementptr %mem[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, f32, f32)>
  llvm.store %other, %f2 : f32, !llvm.ptr
  %loaded = llvm.load %f1 : !llvm.ptr -> f32
  llvm.return %loaded : f32
}

// CHECK-LABEL: llvm.func @gep_other_field(
// CHECK-SAME: %[[VAL:[a-z0-9]+]]: f32
// CHECK-NOT: llvm.load
// CHECK: llvm.return %[[VAL]] : f32

// -----

// An offset that is not known could be the one being read, so the store through
// it stops the forwarding.
llvm.func @gep_unknown_index(%val: f32, %i: i64) -> f32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.array<4 x f32> : (i32) -> !llvm.ptr
  %f0 = llvm.getelementptr %mem[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x f32>
  llvm.store %val, %f0 : f32, !llvm.ptr
  %fi = llvm.getelementptr %mem[0, %i] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x f32>
  llvm.store %val, %fi : f32, !llvm.ptr
  %loaded = llvm.load %f0 : !llvm.ptr -> f32
  llvm.return %loaded : f32
}

// CHECK-LABEL: llvm.func @gep_unknown_index(
// CHECK: %[[LD:.+]] = llvm.load
// CHECK: llvm.return %[[LD]] : f32

// -----

// Two views of one allocation with different shapes reach the same byte by
// different routes: [1, 0] of a 4x4 is element 4, which is [4] of a 16. The
// store through one must not be taken for a different place than the load
// through the other.
llvm.func @cross_shape_same_byte(%val: f32, %other: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1_index = arith.constant 1 : index
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.array<16 x f32> : (i32) -> !llvm.ptr
  %flat = "enzymexla.pointer2memref"(%mem) : (!llvm.ptr) -> memref<16xf32>
  %square = "enzymexla.pointer2memref"(%mem) : (!llvm.ptr) -> memref<4x4xf32>
  memref.store %val, %flat[%c4] : memref<16xf32>
  memref.store %other, %square[%c1_index, %c0] : memref<4x4xf32>
  %loaded = memref.load %flat[%c4] : memref<16xf32>
  llvm.return %loaded : f32
}

// CHECK-LABEL: llvm.func @cross_shape_same_byte(
// CHECK: %[[LD:.+]] = memref.load
// CHECK: llvm.return %[[LD]] : f32

// -----

// Two offsets of the same allocation, counted the same way, add up: the field
// reached by stepping to element 1 and then 2 more is the one reached by
// stepping 3, whether the steps are constant or not.
llvm.func @gep_of_gep(%val: f32, %i: i64) -> f32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c2 = llvm.mlir.constant(2 : i64) : i64
  %c3 = llvm.mlir.constant(3 : i64) : i64
  %mem = llvm.alloca %c1 x !llvm.array<8 x f32> : (i32) -> !llvm.ptr
  %flat = llvm.getelementptr %mem[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x f32>
  %one = llvm.getelementptr %flat[1] : (!llvm.ptr) -> !llvm.ptr, f32
  %three = llvm.getelementptr %one[2] : (!llvm.ptr) -> !llvm.ptr, f32
  llvm.store %val, %three : f32, !llvm.ptr
  %again = llvm.getelementptr %flat[3] : (!llvm.ptr) -> !llvm.ptr, f32
  %loaded = llvm.load %again : !llvm.ptr -> f32
  llvm.return %loaded : f32
}

// CHECK-LABEL: llvm.func @gep_of_gep(
// CHECK-SAME: %[[VAL:[a-z0-9]+]]: f32
// CHECK-NOT: llvm.load
// CHECK: llvm.return %[[VAL]] : f32

// -----

// The same when one of the two steps is not known: %i then 2 is %i + 2, which
// is what the load steps by.
llvm.func @gep_of_gep_symbolic(%val: f32, %i: i64) -> f32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c2 = llvm.mlir.constant(2 : i64) : i64
  %mem = llvm.alloca %c1 x !llvm.array<8 x f32> : (i32) -> !llvm.ptr
  %flat = llvm.getelementptr %mem[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x f32>
  %at = llvm.getelementptr %flat[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %past = llvm.getelementptr %at[2] : (!llvm.ptr) -> !llvm.ptr, f32
  llvm.store %val, %past : f32, !llvm.ptr
  %loaded = llvm.load %past : !llvm.ptr -> f32
  llvm.return %loaded : f32
}

// CHECK-LABEL: llvm.func @gep_of_gep_symbolic(
// CHECK-SAME: %[[VAL:[a-z0-9]+]]: f32
// CHECK-NOT: llvm.load
// CHECK: llvm.return %[[VAL]] : f32

// -----

// Stepping by one value and then another lands where stepping by their sum
// does, which is an expression of both and needs nothing built to name it.
llvm.func @gep_two_values(%val: f32, %i: i64, %j: i64) -> f32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.array<8 x f32> : (i32) -> !llvm.ptr
  %flat = llvm.getelementptr %mem[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x f32>
  %at = llvm.getelementptr %flat[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %past = llvm.getelementptr %at[%j] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %val, %past : f32, !llvm.ptr
  %other = llvm.getelementptr %at[%j] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %loaded = llvm.load %other : !llvm.ptr -> f32
  llvm.return %loaded : f32
}

// CHECK-LABEL: llvm.func @gep_two_values(
// CHECK-SAME: %[[VAL:[a-z0-9]+]]: f32
// CHECK-NOT: llvm.load
// CHECK: llvm.return %[[VAL]] : f32
