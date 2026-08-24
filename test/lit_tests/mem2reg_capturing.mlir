// RUN: enzymexlamlir-opt %s -polygeist-mem2reg -split-input-file | FileCheck %s

llvm.func @llvm_foo_nocapture(%arg0: !llvm.ptr {llvm.nocapture, llvm.readonly}) {
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

// nocapture only says the callee does not hold on to the pointer; an out
// parameter is nocapture and written through before the call returns, so the
// store cannot be forwarded past it. ParMesh::ParMesh passed a Table* slot to
// BuildLocalBoundary(..., Table *&) this way; forwarded, the constructor read
// the stale initial pointer instead of the Table the callee built, and MFEM's
// AMGF solver test died dereferencing it.
llvm.func @llvm_out_param(%arg0: !llvm.ptr {llvm.nocapture})
llvm.func @llvm_store_not_forwarded_past_out_param() -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %val = llvm.mlir.constant(42 : i32) : i32
  llvm.store %val, %mem : i32, !llvm.ptr
  llvm.call @llvm_out_param(%mem) : (!llvm.ptr) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK: llvm.func @llvm_store_not_forwarded_past_out_param() -> i32 {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[AL:.*]] = llvm.alloca %[[C1]] x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK-NEXT: llvm.store %[[C2]], %[[AL]] : i32, !llvm.ptr
// CHECK-NEXT: llvm.call @llvm_out_param(%[[AL]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[LOADED:.*]] = llvm.load %[[AL]] : !llvm.ptr -> i32
// CHECK-NEXT: llvm.return %[[LOADED]] : i32
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

llvm.func @llvm_indirect_call_not_promoted(%fnptr: !llvm.ptr) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %val = llvm.mlir.constant(42 : i32) : i32
  llvm.store %val, %mem : i32, !llvm.ptr
  llvm.call %fnptr(%mem) : !llvm.ptr, (!llvm.ptr) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK: llvm.func @llvm_indirect_call_not_promoted(%[[FN:.*]]: !llvm.ptr) -> i32 {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[AL:.*]] = llvm.alloca %[[C1]] x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK-NEXT: llvm.store %[[C2]], %[[AL]] : i32, !llvm.ptr
// CHECK-NEXT: llvm.call %[[FN]](%[[AL]]) : !llvm.ptr, (!llvm.ptr) -> ()
// CHECK-NEXT: %[[LOADED:.*]] = llvm.load %[[AL]] : !llvm.ptr -> i32
// CHECK-NEXT: llvm.return %[[LOADED]] : i32
// CHECK-NEXT: }

// -----

func.func @func_foo_nocapture(%arg0: !llvm.ptr {llvm.nocapture, llvm.readonly}) {
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

tessera.define @tessera_foo_nocapture(%arg0: !llvm.ptr {llvm.nocapture, llvm.readonly}) attributes {byRefTypes = [!llvm.struct<(i32, i32)>], pure = false} {
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

// -----

llvm.func @variadic_fn(!llvm.ptr, ...)
llvm.func @variadic_call_more_args_than_params() -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %extra = llvm.mlir.constant(100 : i32) : i32
  %val = llvm.mlir.constant(42 : i32) : i32
  llvm.store %val, %mem : i32, !llvm.ptr
  llvm.call @variadic_fn(%mem, %extra, %mem) vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK: llvm.func @variadic_call_more_args_than_params() -> i32 {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[AL:.*]] = llvm.alloca %[[C1]] x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[EXTRA:.*]] = llvm.mlir.constant(100 : i32) : i32
// CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK-NEXT: llvm.store %[[C2]], %[[AL]] : i32, !llvm.ptr
// CHECK-NEXT: llvm.call @variadic_fn(%[[AL]], %[[EXTRA]], %[[AL]]) vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr) -> ()
// CHECK-NEXT: %[[LOADED:.*]] = llvm.load %[[AL]] : !llvm.ptr -> i32
// CHECK-NEXT: llvm.return %[[LOADED]] : i32
// CHECK-NEXT: }

// -----

// A memory-effects attribute on the callee that leaves argument memory and
// other aliasable memory read-only clears the call, and writable
// inaccessible memory does not spoil it: nothing reachable from outside the
// callee can alias the slot through it.
llvm.func @argmem_reader(!llvm.ptr {llvm.nocapture}) attributes {memory_effects = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = readwrite, errnoMem = readwrite, targetMem0 = readwrite, targetMem1 = readwrite>}
llvm.func @forward_across_argmem_read() -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %val = llvm.mlir.constant(42 : i32) : i32
  llvm.store %val, %mem : i32, !llvm.ptr
  llvm.call @argmem_reader(%mem) : (!llvm.ptr) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK: llvm.func @forward_across_argmem_read() -> i32 {
// CHECK: %[[V:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK: llvm.call @argmem_reader
// CHECK: llvm.return %[[V]] : i32

// -----

// Writable argument memory keeps the load.
llvm.func @argmem_writer(!llvm.ptr {llvm.nocapture}) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>}
llvm.func @no_forward_across_argmem_write() -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %val = llvm.mlir.constant(42 : i32) : i32
  llvm.store %val, %mem : i32, !llvm.ptr
  llvm.call @argmem_writer(%mem) : (!llvm.ptr) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK: llvm.func @no_forward_across_argmem_write() -> i32 {
// CHECK: llvm.call @argmem_writer
// CHECK: %[[L:.*]] = llvm.load
// CHECK: llvm.return %[[L]] : i32

// -----

// argmem: read is not enough on its own -- a write classified as other can
// still reach the slot through an alias, so the load stays.
llvm.func @other_writer(!llvm.ptr {llvm.nocapture}) attributes {memory_effects = #llvm.memory_effects<other = readwrite, argMem = read, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>}
llvm.func @no_forward_across_other_write() -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %val = llvm.mlir.constant(42 : i32) : i32
  llvm.store %val, %mem : i32, !llvm.ptr
  llvm.call @other_writer(%mem) : (!llvm.ptr) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK: llvm.func @no_forward_across_other_write() -> i32 {
// CHECK: llvm.call @other_writer
// CHECK: %[[L2:.*]] = llvm.load
// CHECK: llvm.return %[[L2]] : i32

// -----

// The older spelling: a whole-function readonly riding in passthrough.
llvm.func @passthrough_reader(!llvm.ptr {llvm.nocapture}) attributes {passthrough = ["readonly"]}
llvm.func @forward_across_passthrough_readonly() -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %val = llvm.mlir.constant(42 : i32) : i32
  llvm.store %val, %mem : i32, !llvm.ptr
  llvm.call @passthrough_reader(%mem) : (!llvm.ptr) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK: llvm.func @forward_across_passthrough_readonly() -> i32 {
// CHECK: %[[V:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK: llvm.call @passthrough_reader
// CHECK: llvm.return %[[V]] : i32
