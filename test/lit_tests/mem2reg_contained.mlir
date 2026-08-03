// RUN: enzymexlamlir-opt %s -polygeist-mem2reg -split-input-file | FileCheck %s

// A read of part of what an allocation holds is a read of part of the value in
// it: the field is taken out of what was stored, wherever the read spells its
// way there.
func.func @field_of_stored_struct(%val: !llvm.struct<(i64, i32, i32)>) -> i32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.struct<(i64, i32, i32)> : (i32) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%mem) : (!llvm.ptr) -> memref<?x!llvm.struct<(i64, i32, i32)>>
  memref.store %val, %view[%c0] : memref<?x!llvm.struct<(i64, i32, i32)>>
  %punned = "enzymexla.pointer2memref"(%mem) : (!llvm.ptr) -> memref<?xi32>
  %loaded = memref.load %punned[%c2] : memref<?xi32>
  return %loaded : i32
}

// CHECK-LABEL: func.func @field_of_stored_struct(
// CHECK-SAME: %[[VAL:[a-z0-9]+]]: !llvm.struct<(i64, i32, i32)>
// CHECK-NOT: llvm.alloca
// CHECK: %[[F:.+]] = llvm.extractvalue %[[VAL]][1]
// CHECK: return %[[F]] : i32

// -----

// The same through a getelementptr, and for a field that is not the first byte
// of the value it sits in.
llvm.func @field_by_gep(%val: !llvm.struct<(i64, f32, f32)>) -> f32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.struct<(i64, f32, f32)> : (i32) -> !llvm.ptr
  llvm.store %val, %mem : !llvm.struct<(i64, f32, f32)>, !llvm.ptr
  %f2 = llvm.getelementptr %mem[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, f32, f32)>
  %loaded = llvm.load %f2 : !llvm.ptr -> f32
  llvm.return %loaded : f32
}

// CHECK-LABEL: llvm.func @field_by_gep(
// CHECK-SAME: %[[VAL:[a-z0-9]+]]: !llvm.struct<(i64, f32, f32)>
// CHECK-NOT: llvm.alloca
// CHECK: %[[F:.+]] = llvm.extractvalue %[[VAL]][2]
// CHECK: llvm.return %[[F]] : f32

// -----

// An element of an array is reached the same way, and what is nested inside it
// through the way in to that.
llvm.func @element_of_nested_array(%val: !llvm.array<2 x array<2 x i32>>) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.array<2 x array<2 x i32>> : (i32) -> !llvm.ptr
  llvm.store %val, %mem : !llvm.array<2 x array<2 x i32>>, !llvm.ptr
  %at = llvm.getelementptr %mem[0, 1, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x array<2 x i32>>
  %loaded = llvm.load %at : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK-LABEL: llvm.func @element_of_nested_array(
// CHECK-SAME: %[[VAL:[a-z0-9]+]]: !llvm.array<2 x array<2 x i32>>
// CHECK-NOT: llvm.alloca
// CHECK: %[[F:.+]] = llvm.extractvalue %[[VAL]][1, 1]
// CHECK: llvm.return %[[F]] : i32

// -----

// A read that straddles two fields is not any one of them, so there is no way
// into the value that reaches it and the allocation stays.
llvm.func @straddling_read(%val: !llvm.struct<(i32, i32, i32, i32)>) -> i64 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.struct<(i32, i32, i32, i32)> : (i32) -> !llvm.ptr
  llvm.store %val, %mem : !llvm.struct<(i32, i32, i32, i32)>, !llvm.ptr
  %mid = llvm.getelementptr %mem[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32)>
  %loaded = llvm.load %mid : !llvm.ptr -> i64
  llvm.return %loaded : i64
}

// CHECK-LABEL: llvm.func @straddling_read(
// CHECK: %[[LD:.+]] = llvm.load
// CHECK: llvm.return %[[LD]] : i64

// -----

// A write of a part says nothing of the rest, so what was there before it does
// not reach a read after it.
llvm.func @write_of_a_part(%val: !llvm.struct<(i32, i32)>, %other: i32) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.struct<(i32, i32)> : (i32) -> !llvm.ptr
  llvm.store %val, %mem : !llvm.struct<(i32, i32)>, !llvm.ptr
  %f1 = llvm.getelementptr %mem[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
  llvm.store %other, %f1 : i32, !llvm.ptr
  %f0 = llvm.getelementptr %mem[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
  %loaded = llvm.load %f0 : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK-LABEL: llvm.func @write_of_a_part(
// CHECK: %[[LD:.+]] = llvm.load
// CHECK: llvm.return %[[LD]] : i32
