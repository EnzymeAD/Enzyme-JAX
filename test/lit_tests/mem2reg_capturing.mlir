// RUN: enzymexlamlir-opt %s -polygeist-mem2reg -split-input-file | FileCheck %s

llvm.func @llvm_foo_nocapture(%arg0: !llvm.ptr {llvm.nocapture}) {
  llvm.return
}
llvm.func @llvm_store_to_load_forwarded() -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %val = llvm.mlir.constant(42 : i32) : i32
  llvm.store %val, %mem : i32, !llvm.ptr
  llvm.call @llvm_foo_nocapture(%mem) : (!llvm.ptr) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK: llvm.func @llvm_store_to_load_forwarded() -> i32 {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[AL:.*]] = llvm.alloca %[[C1]] x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK-NEXT: llvm.store %[[C2]], %[[AL]] : i32, !llvm.ptr
// CHECK-NEXT: llvm.call @llvm_foo_nocapture(%[[AL]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: llvm.return %[[C2]] : i32
// CHECK-NEXT: }

// -----

llvm.func @llvm_foo_capturing(%arg0: !llvm.ptr) {
  llvm.return
}
llvm.func @llvm_store_to_load_not_forwarded() -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %val = llvm.mlir.constant(42 : i32) : i32
  llvm.store %val, %mem : i32, !llvm.ptr
  llvm.call @llvm_foo_capturing(%mem) : (!llvm.ptr) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK: llvm.func @llvm_store_to_load_not_forwarded() -> i32 {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[AL:.*]] = llvm.alloca %[[C1]] x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK-NEXT: llvm.store %[[C2]], %[[AL]] : i32, !llvm.ptr
// CHECK-NEXT: llvm.call @llvm_foo_capturing(%[[AL]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[LOADED:.*]] = llvm.load %[[AL]] : !llvm.ptr -> i32
// CHECK-NEXT: llvm.return %[[LOADED]] : i32
// CHECK-NEXT: }

// -----

func.func @func_foo_nocapture(%arg0: !llvm.ptr {llvm.nocapture}) {
  func.return
}
func.func @func_store_to_load_forwarded() -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %val = llvm.mlir.constant(42 : i32) : i32
  llvm.store %val, %mem : i32, !llvm.ptr
  func.call @func_foo_nocapture(%mem) : (!llvm.ptr) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  func.return %loaded : i32
}

// CHECK: func.func @func_store_to_load_forwarded() -> i32 {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[AL:.*]] = llvm.alloca %[[C1]] x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK-NEXT: llvm.store %[[C2]], %[[AL]] : i32, !llvm.ptr
// CHECK-NEXT: call @func_foo_nocapture(%[[AL]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: return %[[C2]] : i32
// CHECK-NEXT: }

// -----

func.func @func_foo_capturing(%arg0: !llvm.ptr) {
  func.return
}
func.func @func_store_to_load_not_forwarded() -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %val = llvm.mlir.constant(42 : i32) : i32
  llvm.store %val, %mem : i32, !llvm.ptr
  func.call @func_foo_capturing(%mem) : (!llvm.ptr) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  func.return %loaded : i32
}

// CHECK: func.func @func_store_to_load_not_forwarded() -> i32 {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[AL:.*]] = llvm.alloca %[[C1]] x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK-NEXT: llvm.store %[[C2]], %[[AL]] : i32, !llvm.ptr
// CHECK-NEXT: call @func_foo_capturing(%[[AL]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[LOADED:.*]] = llvm.load %[[AL]] : !llvm.ptr -> i32
// CHECK-NEXT: return %[[LOADED]] : i32
// CHECK-NEXT: }

// -----

tessera.define @tessera_foo_nocapture(%arg0: !llvm.ptr {llvm.nocapture}) attributes {byRefTypes = [!llvm.struct<(i32, i32)>], pure = false} {
  tessera.return
}
tessera.define @tessera_store_to_load_forwarded() -> i32 attributes {byRefTypes = [], pure = false} {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %val = llvm.mlir.constant(42 : i32) : i32
  llvm.store %val, %mem : i32, !llvm.ptr
  tessera.call @tessera_foo_nocapture(%mem) : (!llvm.ptr) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  tessera.return %loaded : i32
}

// CHECK: tessera.define @tessera_store_to_load_forwarded() -> i32 attributes {byRefTypes = [], pure = false} {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[AL:.*]] = llvm.alloca %[[C1]] x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK-NEXT: llvm.store %[[C2]], %[[AL]] : i32, !llvm.ptr
// CHECK-NEXT: tessera.call @tessera_foo_nocapture(%[[AL]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: tessera.return %[[C2]] : i32
// CHECK-NEXT: }

// -----

tessera.define @tessera_foo_capturing(%arg0: !llvm.ptr) attributes {byRefTypes = [!llvm.struct<(i32, i32)>], pure = false} {
  tessera.return
}
tessera.define @tessera_store_to_load_not_forwarded() -> i32 attributes {byRefTypes = [], pure = false} {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %val = llvm.mlir.constant(42 : i32) : i32
  llvm.store %val, %mem : i32, !llvm.ptr
  tessera.call @tessera_foo_capturing(%mem) : (!llvm.ptr) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  tessera.return %loaded : i32
}

// CHECK: tessera.define @tessera_store_to_load_not_forwarded() -> i32 attributes {byRefTypes = [], pure = false} {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[AL:.*]] = llvm.alloca %[[C1]] x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK-NEXT: llvm.store %[[C2]], %[[AL]] : i32, !llvm.ptr
// CHECK-NEXT: tessera.call @tessera_foo_capturing(%[[AL]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[LOADED:.*]] = llvm.load %[[AL]] : !llvm.ptr -> i32
// CHECK-NEXT: tessera.return %[[LOADED]] : i32
// CHECK-NEXT: }