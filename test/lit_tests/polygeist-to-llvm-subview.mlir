// RUN: enzymexlamlir-opt %s --convert-polygeist-to-llvm -split-input-file | FileCheck %s

// A subview is a GEP of the base pointer by its offsets; its contiguous
// strided layout lowers like the identity one.

// CHECK-LABEL: llvm.func @dynamic_offset(
// CHECK-SAME: %[[BASE:.+]]: !llvm.ptr, %[[OFF:.+]]: i64, %[[I:.+]]: i64)
// CHECK: %[[VIEW:.+]] = llvm.getelementptr %[[BASE]][%[[OFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[PTR:.+]] = llvm.getelementptr %[[VIEW]][%[[I]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.load %[[PTR]] : !llvm.ptr -> f32
func.func @dynamic_offset(%arg0: memref<?xf32>, %off: index, %i: index) -> f32 {
  %0 = memref.subview %arg0[%off] [4] [1] : memref<?xf32> to memref<4xf32, strided<[1], offset: ?>>
  %1 = memref.load %0[%i] : memref<4xf32, strided<[1], offset: ?>>
  return %1 : f32
}

// -----

// CHECK-LABEL: llvm.func @rows(
// CHECK-SAME: %[[BASE:.+]]: !llvm.ptr, %[[I:.+]]: i64, %[[J:.+]]: i64)
// CHECK: %[[VIEW:.+]] = llvm.getelementptr %[[BASE]][1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x f32>
// CHECK: %[[PTR:.+]] = llvm.getelementptr %[[VIEW]][%[[I]], %[[J]]] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<8 x f32>
// CHECK: llvm.load %[[PTR]] : !llvm.ptr -> f32
func.func @rows(%arg0: memref<4x8xf32>, %i: index, %j: index) -> f32 {
  %0 = memref.subview %arg0[1, 0] [2, 8] [1, 1] : memref<4x8xf32> to memref<2x8xf32, strided<[8, 1], offset: 8>>
  %1 = memref.load %0[%i, %j] : memref<2x8xf32, strided<[8, 1], offset: 8>>
  return %1 : f32
}
