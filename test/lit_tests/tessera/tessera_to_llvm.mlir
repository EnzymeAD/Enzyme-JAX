// RUN: enzymexlamlir-opt %s -tessera-to-llvm -split-input-file | FileCheck %s

tessera.define @tessera_simple_func() attributes {argModes = [], pure = false, tessera.original_name = "simple_func"} {
  tessera.return
}

// CHECK: llvm.func @simple_func()
// CHECK-NEXT: llvm.return

// -----

tessera.define @tessera_func_with_args(%arg0: i32, %arg1: f32) -> i32 attributes {argModes = [unit, unit], pure = false, tessera.original_name = "func_with_args"} {
  tessera.return %arg0 : i32
}

// CHECK: llvm.func @func_with_args(%arg0: i32, %arg1: f32) -> i32
// CHECK-NEXT: llvm.return %arg0 : i32

// -----

tessera.define @tessera_helper() attributes {argModes = [], pure = false, tessera.original_name = "helper"} {
  tessera.return
}

tessera.define @tessera_func_with_call() attributes {argModes = [], pure = false, tessera.original_name = "func_with_call"} {
  tessera.call @tessera_helper() {op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 0, 0>} : () -> ()
  tessera.return
}

// CHECK: llvm.func @helper
// CHECK-NEXT: llvm.return

// CHECK: llvm.func @func_with_call
// CHECK-NEXT: llvm.call @helper() : () -> ()
// CHECK-NEXT: llvm.return

// -----

// An op only declared in the file comes after its callers, as clang emits
// declarations last. Its calls still convert, and name it by its original name.
llvm.func @calls_declared(%x: i32) -> i32 {
  %r = tessera.call @tessera_declared(%x) {op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>} : (i32) -> i32
  llvm.return %r : i32
}

tessera.define private @tessera_declared(i32) -> i32 attributes {argModes = [unit], pure = false, tessera.original_name = "declared"}

// CHECK-LABEL: llvm.func @calls_declared(
// CHECK-NEXT: %[[R:.*]] = llvm.call @declared(%arg0) : (i32) -> i32
// CHECK-NEXT: llvm.return %[[R]] : i32
// CHECK: llvm.func {{.*}}@declared(i32) -> i32
// CHECK-NOT: tessera.

// -----

tessera.define @tessera_sret_func(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.nonnull, llvm.sret = !llvm.struct<(f32, f32)>}, %arg1: !llvm.ptr {llvm.noundef, llvm.readonly}) 
attributes {argModes = [{dir = #tessera.dir<in>, type = !llvm.struct<(f32, f32)>}], linkage = #llvm.linkage<external>, pure = true, tessera.original_name = "sret_func"} {
  %0 = llvm.load %arg1 <alignment = 8> : !llvm.ptr -> f32
  llvm.store %0, %arg0 <alignment = 8> : f32, !llvm.ptr
  tessera.return
}

llvm.func @caller() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %0 x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.load %2 : !llvm.ptr -> !llvm.struct<(f32, f32)>
  %4 = tessera.call @tessera_sret_func(%3) <{arg_attrs = [{llvm.nonnull, llvm.noundef}]}> {op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 2, 0>} : (!llvm.struct<(f32, f32)>) -> !llvm.struct<(f32, f32)>
  llvm.store %4, %1 : !llvm.struct<(f32, f32)>, !llvm.ptr
  llvm.return
}

// CHECK: llvm.func @sret_func(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.nonnull, llvm.sret = !llvm.struct<(f32, f32)>}, %arg1: !llvm.ptr {llvm.noundef, llvm.readonly})
// CHECK-NEXT: %[[LOAD:.*]] = llvm.load %arg1 <alignment = 8> : !llvm.ptr -> f32
// CHECK-NEXT: llvm.store %[[LOAD]], %arg0 <alignment = 8> : f32, !llvm.ptr
// CHECK-NEXT: llvm.return

// CHECK: llvm.func @caller()
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[A1:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// The argument is passed where it was loaded from, as the original call did.
// CHECK-NEXT: %[[A2:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[SRET:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(f32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT: llvm.call @sret_func(%[[SRET]], %[[A2]]) : (!llvm.ptr {llvm.align = 8 : i64, llvm.nonnull, llvm.sret = !llvm.struct<(f32, f32)>}, !llvm.ptr {llvm.nonnull, llvm.noundef}) -> ()
// CHECK-NEXT: %[[LOADED:.*]] = llvm.load %[[SRET]] : !llvm.ptr -> !llvm.struct<(f32, f32)>
// CHECK-NEXT: llvm.store %[[LOADED]], %[[A1]] : !llvm.struct<(f32, f32)>, !llvm.ptr
// CHECK-NEXT: llvm.return

// -----

tessera.define @tessera_func_with_result_arg(%arg0: !llvm.ptr, %arg1: i32) -> i32 attributes {argModes = [{dir = #tessera.dir<out>, type = i32}, unit], pure = false, tessera.original_name = "func_with_result_arg"} {
  llvm.store %arg1, %arg0 : i32, !llvm.ptr
  tessera.return %arg1 : i32
}

llvm.func @result_arg_func_caller() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  %2:2 = tessera.call @tessera_func_with_result_arg(%0) : (i32) -> (i32, i32)
  llvm.store %2#0, %1 : i32, !llvm.ptr
  llvm.return
}

// CHECK: llvm.func @func_with_result_arg(%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: i32) -> i32 {
// CHECK-NEXT: llvm.store %[[ARG1]], %[[ARG0]] : i32, !llvm.ptr
// CHECK-NEXT: llvm.return %[[ARG1]] : i32
// CHECK-NEXT: }

// CHECK: llvm.func @result_arg_func_caller() {
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[AL:.*]] = llvm.alloca %[[ONE]] x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[AL2:.*]] = llvm.alloca %[[ONE]] x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[RES:.*]] = llvm.call @func_with_result_arg(%[[AL2]], %[[ONE]]) : (!llvm.ptr, i32) -> i32
// CHECK-NEXT: %[[LOAD:.*]] = llvm.load %[[AL2]] : !llvm.ptr -> i32
// CHECK-NEXT: llvm.store %[[LOAD]], %[[AL]] : i32, !llvm.ptr
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }
// -----

// A tail call promises the callee does not access the caller's allocas. Once
// the call is given stack copies of its lifted arguments that no longer holds,
// so the marker is dropped: kept, LLVM would assume the callee leaves the copy
// of the inout argument alone and fold the value read back after the call to
// the one stored before it, losing what the callee wrote.

tessera.define private @tessera_update(!llvm.ptr) -> i32 attributes {argModes = [{dir = #tessera.dir<inout>, type = !llvm.struct<(i64, i32)>}], pure = false, tessera.original_name = "update"}
tessera.define private @tessera_plain(i64) -> i32 attributes {argModes = [unit], pure = false, tessera.original_name = "plain"}

llvm.func @tail_caller(%p: !llvm.ptr, %n: i64) -> i32 {
  %v = llvm.mlir.zero : !llvm.struct<(i64, i32)>
  %r:2 = tessera.call @tessera_update(%v) {TailCallKind = #llvm.tailcallkind<tail>} : (!llvm.struct<(i64, i32)>) -> (!llvm.struct<(i64, i32)>, i32)
  llvm.store %r#0, %p : !llvm.struct<(i64, i32)>, !llvm.ptr
  %s = tessera.call @tessera_plain(%n) {TailCallKind = #llvm.tailcallkind<tail>} : (i64) -> i32
  %w = llvm.load %p : !llvm.ptr -> !llvm.struct<(i64, i32)>
  %t:2 = tessera.call @tessera_update(%w) {TailCallKind = #llvm.tailcallkind<tail>} : (!llvm.struct<(i64, i32)>) -> (!llvm.struct<(i64, i32)>, i32)
  llvm.store %t#0, %p : !llvm.struct<(i64, i32)>, !llvm.ptr
  llvm.return %s : i32
}

// CHECK-LABEL: llvm.func @tail_caller
// CHECK: %[[COPY:.*]] = llvm.alloca
// CHECK: llvm.call @update(%[[COPY]]) : (!llvm.ptr) -> i32
// CHECK: llvm.load %[[COPY]]
// A call given no stack copy keeps its marker: one with no pointer operand,
// and one given the address its operand was loaded from, as the call it came
// from was.
// CHECK: llvm.call tail @plain(%arg1)
// CHECK: llvm.call tail @update(%arg0)

// -----

// A value passed at two positions was loaded through one pointer, as in
// mpfi_mul(r, r, y), so both get the same stack copy: the callee sees the
// operands alias, as it would have in the original call. Two copies would
// share the limbs they point to without sharing the storage, and a callee that
// tests for aliasing by comparing pointers would miss it.

!interval = !llvm.struct<(i64, i32, i64, ptr)>

tessera.define private @tessera_mul(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [{dir = #tessera.dir<inout>, type = !interval}, {dir = #tessera.dir<in>, type = !interval}, {dir = #tessera.dir<in>, type = !interval}], pure = false, tessera.original_name = "mul"}

llvm.func @aliased_caller(%p: !llvm.ptr, %q: !llvm.ptr) -> i32 {
  %r = llvm.mlir.zero : !interval
  %y = llvm.mlir.undef : !interval
  %0:2 = tessera.call @tessera_mul(%r, %r, %y) : (!interval, !interval, !interval) -> (!interval, i32)
  llvm.store %0#0, %p : !interval, !llvm.ptr
  llvm.return %0#1 : i32
}

// CHECK-LABEL: llvm.func @aliased_caller
// CHECK: %[[R:.*]] = llvm.alloca
// CHECK-NEXT: llvm.store %{{.*}}, %[[R]]
// CHECK-NEXT: %[[Y:.*]] = llvm.alloca
// CHECK-NEXT: llvm.store %{{.*}}, %[[Y]]
// CHECK-NEXT: %[[RES:.*]] = llvm.call @mul(%[[R]], %[[R]], %[[Y]])
// CHECK-NEXT: %[[OUT:.*]] = llvm.load %[[R]]
// CHECK-NEXT: llvm.store %[[OUT]], %arg0

// -----

// An operand the callee takes by pointer is passed the address it was loaded
// from, as the call it came from was, when nothing that runs in between may
// write memory. Operands that alias then alias in the call however they were
// computed: mpfi_mul(r, x, y) with r and x the same interval reached through
// different pointers still passes one pointer twice. Anything that might write
// in between gets a copy instead.

!interval = !llvm.struct<(i64, i32, i64, ptr)>

tessera.define private @tessera_mul(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [{dir = #tessera.dir<inout>, type = !interval}, {dir = #tessera.dir<in>, type = !interval}, {dir = #tessera.dir<in>, type = !interval}], pure = false, tessera.original_name = "mul"}
tessera.define private @tessera_is_empty(!llvm.ptr) -> i32 attributes {argModes = [{dir = #tessera.dir<in>, type = !interval}], pure = true, tessera.original_name = "is_empty"}
llvm.func @clobber(!llvm.ptr)

// CHECK-LABEL: llvm.func @same_memory
// CHECK: llvm.call @mul(%arg0, %arg1, %arg2)
// CHECK-NEXT: llvm.load %arg0
llvm.func @same_memory(%p: !llvm.ptr, %p2: !llvm.ptr, %q: !llvm.ptr) -> i32 {
  %r = llvm.load %p : !llvm.ptr -> !interval
  %x = llvm.load %p2 : !llvm.ptr -> !interval
  %y = llvm.load %q : !llvm.ptr -> !interval
  %0:2 = tessera.call @tessera_mul(%r, %x, %y) : (!interval, !interval, !interval) -> (!interval, i32)
  llvm.store %0#0, %p : !interval, !llvm.ptr
  llvm.return %0#1 : i32
}

// A guard's check runs between the loads and the call it guards; it writes
// none of its arguments, so the addresses still hold the values.
// CHECK-LABEL: llvm.func @after_a_check
// CHECK: llvm.call @is_empty(%arg0)
// CHECK: scf.if
// CHECK: llvm.call @mul(%arg0, %arg0, %arg1)
llvm.func @after_a_check(%p: !llvm.ptr, %q: !llvm.ptr) -> i32 {
  %r = llvm.load %p : !llvm.ptr -> !interval
  %y = llvm.load %q : !llvm.ptr -> !interval
  %e = tessera.call @tessera_is_empty(%r) : (!interval) -> i32
  %z = llvm.mlir.constant(0 : i32) : i32
  %c = llvm.icmp "eq" %e, %z : i32
  %0:2 = scf.if %c -> (!interval, i32) {
    %1:2 = tessera.call @tessera_mul(%r, %r, %y) : (!interval, !interval, !interval) -> (!interval, i32)
    scf.yield %1#0, %1#1 : !interval, i32
  } else {
    scf.yield %r, %z : !interval, i32
  }
  llvm.store %0#0, %p : !interval, !llvm.ptr
  llvm.return %0#1 : i32
}

// A call that might write memory in between: the loaded value is copied.
// CHECK-LABEL: llvm.func @after_a_write
// CHECK: llvm.call @clobber
// CHECK: %[[C:.*]] = llvm.alloca
// CHECK: %[[Y:.*]] = llvm.alloca
// CHECK: llvm.call @mul(%[[C]], %[[C]], %[[Y]])
llvm.func @after_a_write(%p: !llvm.ptr, %q: !llvm.ptr) -> i32 {
  %r = llvm.load %p : !llvm.ptr -> !interval
  %y = llvm.load %q : !llvm.ptr -> !interval
  llvm.call @clobber(%p) : (!llvm.ptr) -> ()
  %0:2 = tessera.call @tessera_mul(%r, %r, %y) : (!interval, !interval, !interval) -> (!interval, i32)
  llvm.store %0#0, %p : !interval, !llvm.ptr
  llvm.return %0#1 : i32
}

// Loaded outside a loop and passed inside it: the call runs again after what
// follows it in the body, so the loaded value is copied.
// CHECK-LABEL: llvm.func @across_a_loop
// CHECK: scf.for
// CHECK: %[[C:.*]] = llvm.alloca
// CHECK: %[[Y:.*]] = llvm.alloca
// CHECK: llvm.call @mul(%[[C]], %[[C]], %[[Y]])
llvm.func @across_a_loop(%p: !llvm.ptr, %q: !llvm.ptr, %n: i64) {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %r = llvm.load %p : !llvm.ptr -> !interval
  %y = llvm.load %q : !llvm.ptr -> !interval
  scf.for %i = %c0 to %n step %c1 : i64 {
    %0:2 = tessera.call @tessera_mul(%r, %r, %y) : (!interval, !interval, !interval) -> (!interval, i32)
    llvm.store %0#0, %p : !interval, !llvm.ptr
  }
  llvm.return
}
