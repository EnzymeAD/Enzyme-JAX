// RUN: enzymexlamlir-opt %s -tessera-propagate-properties -split-input-file | FileCheck %s

// `tessera.assumes` on a pointer parameter, as `IntVec &ids` with
// tessera_assumes("sorted(ids)"), is about the object it points to when the
// function starts. The function may change that object, so the fact holds at a
// call only if the walk back from the call reaches the function's start
// without passing anything that might change it. A caller's pointers may reach
// the object too, so that includes calls and stores that don't name it.

module {
  tessera.define private @intvec.contains(!llvm.ptr {tessera.readonly}, i32) -> i1 attributes {argModes = [unit, unit], pure = false}
  llvm.func @push_back(!llvm.ptr, i32)
  llvm.func @remove_if(!llvm.ptr, !llvm.ptr) attributes {tessera.preserves = [{inputs = [0], property = "sorted"}]}
  llvm.func @log_something()
  llvm.func @reads_only(i32) -> i32 attributes {memory_effects = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>}

  // On entry, and still after a readonly call.
  // CHECK-LABEL: llvm.func @on_entry
  // CHECK: tessera.call @intvec.contains({{.*}}) <arg_attrs = [{tessera.property = ["sorted"]}, {}]>
  // CHECK: tessera.call @intvec.contains({{.*}}) <arg_attrs = [{tessera.property = ["sorted"]}, {}]>
  llvm.func @on_entry(%v: !llvm.ptr {tessera.assumes = ["sorted"]}, %x: i32) -> i1 {
    %a = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
    %b = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
    %r = llvm.and %a, %b : i1
    llvm.return %r : i1
  }

  // Holds before the function changes the object, and not after.
  // CHECK-LABEL: llvm.func @changed_after_entry
  // CHECK: tessera.call @intvec.contains({{.*}}) <arg_attrs = [{tessera.property = ["sorted"]}, {}]>
  // CHECK: llvm.call @push_back
  // CHECK: tessera.call @intvec.contains({{[^)]*}}) :
  llvm.func @changed_after_entry(%v: !llvm.ptr {tessera.assumes = ["sorted"]}, %x: i32) -> i1 {
    %a = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
    llvm.call @push_back(%v, %x) : (!llvm.ptr, i32) -> ()
    %b = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
    %r = llvm.and %a, %b : i1
    llvm.return %r : i1
  }

  // A call that changes the object but preserves the property keeps it.
  // CHECK-LABEL: llvm.func @preserved
  // CHECK: tessera.call @intvec.contains({{.*}}) <arg_attrs = [{tessera.property = ["sorted"]}, {}]>
  llvm.func @preserved(%v: !llvm.ptr {tessera.assumes = ["sorted"]}, %p: !llvm.ptr, %x: i32) -> i1 {
    llvm.call @remove_if(%v, %p) : (!llvm.ptr, !llvm.ptr) -> ()
    %r = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
    llvm.return %r : i1
  }

  // A call not given the object might still reach it, through a pointer the
  // caller kept.
  // CHECK-LABEL: llvm.func @unknown_call
  // CHECK: tessera.call @intvec.contains({{[^)]*}}) :
  llvm.func @unknown_call(%v: !llvm.ptr {tessera.assumes = ["sorted"]}, %x: i32) -> i1 {
    llvm.call @log_something() : () -> ()
    %r = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
    llvm.return %r : i1
  }

  // Unless it writes no memory.
  // CHECK-LABEL: llvm.func @call_without_writes
  // CHECK: tessera.call @intvec.contains({{.*}}) <arg_attrs = [{tessera.property = ["sorted"]}, {}]>
  llvm.func @call_without_writes(%v: !llvm.ptr {tessera.assumes = ["sorted"]}, %x: i32) -> i1 {
    %y = llvm.call @reads_only(%x) : (i32) -> i32
    %r = tessera.call @intvec.contains(%v, %y) : (!llvm.ptr, i32) -> i1
    llvm.return %r : i1
  }

  // A store to the function's own stack can't reach it.
  // CHECK-LABEL: llvm.func @local_store
  // CHECK: tessera.call @intvec.contains({{.*}}) <arg_attrs = [{tessera.property = ["sorted"]}, {}]>
  llvm.func @local_store(%v: !llvm.ptr {tessera.assumes = ["sorted"]}, %x: i32) -> i1 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
    llvm.store %x, %slot : i32, !llvm.ptr
    %r = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
    llvm.return %r : i1
  }

  // A store through another parameter might.
  // CHECK-LABEL: llvm.func @store_through_other
  // CHECK: tessera.call @intvec.contains({{[^)]*}}) :
  llvm.func @store_through_other(%v: !llvm.ptr {tessera.assumes = ["sorted"]}, %w: !llvm.ptr, %x: i32) -> i1 {
    llvm.store %x, %w : i32, !llvm.ptr
    %r = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
    llvm.return %r : i1
  }

  // In a loop that only reads it.
  // CHECK-LABEL: llvm.func @readonly_loop
  // CHECK: scf.for
  // CHECK: tessera.call @intvec.contains({{.*}}) <arg_attrs = [{tessera.property = ["sorted"]}, {}]>
  llvm.func @readonly_loop(%v: !llvm.ptr {tessera.assumes = ["sorted"]}, %x: i32, %count: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.index_cast %count : i64 to index
    scf.for %i = %c0 to %n step %c1 {
      %r = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
    }
    llvm.return
  }

  // In a loop that changes it, the next iteration's call sees the change.
  // CHECK-LABEL: llvm.func @changing_loop
  // CHECK: scf.for
  // CHECK: tessera.call @intvec.contains({{[^)]*}}) :
  llvm.func @changing_loop(%v: !llvm.ptr {tessera.assumes = ["sorted"]}, %x: i32, %count: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.index_cast %count : i64 to index
    scf.for %i = %c0 to %n step %c1 {
      %r = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
      llvm.call @push_back(%v, %x) : (!llvm.ptr, i32) -> ()
    }
    llvm.return
  }
}

// -----

// What isn't a pointer can't change, so an assumption on it holds throughout.
// And a value loaded from a pointer parameter gets the assumption if the load
// read the object as it was on entry.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.shift(i64) -> i64 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.use(!llvm.ptr) -> f64 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}
  llvm.func @log_something()

  // CHECK-LABEL: llvm.func @scalar
  // CHECK: tessera.call @lib.shift(%{{.*}}) <arg_attrs = [{tessera.property = ["power_of_two"]}]>
  llvm.func @scalar(%n: i64 {tessera.assumes = ["power_of_two"]}) -> i64 {
    llvm.call @log_something() : () -> ()
    %r = tessera.call @lib.shift(%n) : (i64) -> i64
    llvm.return %r : i64
  }

  // Loaded before the call that might change it.
  // CHECK-LABEL: llvm.func @loaded_on_entry
  // CHECK: tessera.call @lib.use(%{{.*}}) <arg_attrs = [{tessera.property = ["SPD"]}]>
  llvm.func @loaded_on_entry(%M: !llvm.ptr {tessera.assumes = ["SPD"]}) -> f64 {
    %m = llvm.load %M : !llvm.ptr -> !mat
    llvm.call @log_something() : () -> ()
    %r = tessera.call @lib.use(%m) : (!mat) -> f64
    llvm.return %r : f64
  }

  // Loaded after.
  // CHECK-LABEL: llvm.func @loaded_after_change
  // CHECK: tessera.call @lib.use(%{{[^)]*}}) :
  llvm.func @loaded_after_change(%M: !llvm.ptr {tessera.assumes = ["SPD"]}) -> f64 {
    llvm.call @log_something() : () -> ()
    %m = llvm.load %M : !llvm.ptr -> !mat
    %r = tessera.call @lib.use(%m) : (!mat) -> f64
    llvm.return %r : f64
  }
}
