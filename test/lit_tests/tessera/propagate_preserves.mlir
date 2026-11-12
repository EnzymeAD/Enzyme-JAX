// RUN: enzymexlamlir-opt %s -tessera-propagate-properties -split-input-file | FileCheck %s

// A preserves is the conditional guarantee whose condition is itself:
// tessera_preserves("sorted(this)") lifts to a conditional establishes of
// sorted on `this` if sorted(this). So a call that changes the object, but
// keeps it sorted, passes the fact on when it held before the call.

module {
  tessera.define private @intvec.contains(!llvm.ptr {tessera.unchanged}, i32) -> i1 attributes {argModes = [unit, unit], pure = false}
  llvm.func @remove_if(!llvm.ptr, !llvm.ptr) attributes {tessera.conditional_establishes = [{arg = 0 : i64, condition = "sorted(this)", constants = {}, names = {this = 0 : i64}, property = "sorted"}]}
  // Returns a value, which a preserves of `this` says nothing about.
  llvm.func @push_sorted(!llvm.ptr, i32) -> i32 attributes {tessera.conditional_establishes = [{arg = 0 : i64, condition = "sorted(this)", constants = {}, names = {this = 0 : i64}, property = "sorted"}]}
  // Keeps the vector sorted only if n is positive.
  llvm.func @truncate(!llvm.ptr, i32) attributes {tessera.conditional_establishes = [{arg = 0 : i64, condition = "sorted(this) && (n > 0)", constants = {}, names = {n = 1 : i64, this = 0 : i64}, property = "sorted"}]}
  // Makes the vector unique, and keeps it sorted.
  llvm.func @dedupe(!llvm.ptr {tessera.establishes = ["unique"]}) attributes {tessera.conditional_establishes = [{arg = 0 : i64, condition = "sorted(this)", constants = {}, names = {this = 0 : i64}, property = "sorted"}]}
  // A tessera op that preserves sorted.
  tessera.define private @intvec.remove_value(!llvm.ptr, i32) attributes {argModes = [unit, unit], pure = false, tessera.conditional_establishes = [{arg = 0 : i64, condition = "sorted(this)", constants = {}, names = {this = 0 : i64}, property = "sorted"}]}

  // Through two preserving calls in a row.
  // CHECK-LABEL: llvm.func @chain
  // CHECK: tessera.call @intvec.contains({{.*}}) <arg_attrs = [{tessera.property = ["sorted"]}, {}]>
  llvm.func @chain(%v: !llvm.ptr {tessera.assumes = ["sorted"]}, %p: !llvm.ptr, %x: i32) -> i1 {
    llvm.call @remove_if(%v, %p) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @remove_if(%v, %p) : (!llvm.ptr, !llvm.ptr) -> ()
    %r = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
    llvm.return %r : i1
  }

  // Nothing to preserve: the object was not known to be sorted.
  // CHECK-LABEL: llvm.func @not_sorted_before
  // CHECK: tessera.call @intvec.contains({{[^)]*}}) :
  llvm.func @not_sorted_before(%v: !llvm.ptr, %p: !llvm.ptr, %x: i32) -> i1 {
    llvm.call @remove_if(%v, %p) : (!llvm.ptr, !llvm.ptr) -> ()
    %r = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
    llvm.return %r : i1
  }

  // A function that returns a value can preserve a property of an argument.
  // CHECK-LABEL: llvm.func @returns_value
  // CHECK: tessera.call @intvec.contains({{.*}}) <arg_attrs = [{tessera.property = ["sorted"]}, {}]>
  llvm.func @returns_value(%v: !llvm.ptr {tessera.assumes = ["sorted"]}, %x: i32) -> i1 {
    %n = llvm.call @push_sorted(%v, %x) : (!llvm.ptr, i32) -> i32
    %r = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
    llvm.return %r : i1
  }

  // The preserves' own condition holds.
  // CHECK-LABEL: llvm.func @condition_holds
  // CHECK: tessera.call @intvec.contains({{.*}}) <arg_attrs = [{tessera.property = ["sorted"]}, {}]>
  llvm.func @condition_holds(%v: !llvm.ptr {tessera.assumes = ["sorted"]}, %x: i32) -> i1 {
    %c3 = llvm.mlir.constant(3 : i32) : i32
    llvm.call @truncate(%v, %c3) : (!llvm.ptr, i32) -> ()
    %r = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
    llvm.return %r : i1
  }

  // It doesn't, so the call is like any other that changes the object.
  // CHECK-LABEL: llvm.func @condition_fails
  // CHECK: tessera.call @intvec.contains({{[^)]*}}) :
  llvm.func @condition_fails(%v: !llvm.ptr {tessera.assumes = ["sorted"]}, %x: i32) -> i1 {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    llvm.call @truncate(%v, %c0) : (!llvm.ptr, i32) -> ()
    %r = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
    llvm.return %r : i1
  }

  // What a call establishes outright and what it preserves both hold after
  // it.
  // CHECK-LABEL: llvm.func @establishes_and_preserves
  // CHECK: tessera.call @intvec.contains({{.*}}) <arg_attrs = [{tessera.property = ["sorted", "unique"]}, {}]>
  llvm.func @establishes_and_preserves(%v: !llvm.ptr {tessera.assumes = ["sorted"]}, %x: i32) -> i1 {
    llvm.call @dedupe(%v) : (!llvm.ptr) -> ()
    %r = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
    llvm.return %r : i1
  }

  // A tessera op preserves a property as a plain function does.
  // CHECK-LABEL: llvm.func @tessera_op_preserves
  // CHECK: tessera.call @intvec.contains({{.*}}) <arg_attrs = [{tessera.property = ["sorted"]}, {}]>
  llvm.func @tessera_op_preserves(%v: !llvm.ptr {tessera.assumes = ["sorted"]}, %x: i32) -> i1 {
    tessera.call @intvec.remove_value(%v, %x) : (!llvm.ptr, i32) -> ()
    %r = tessera.call @intvec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
    llvm.return %r : i1
  }
}

// -----

// A conditional guarantee on a value: the result of a call has the property
// where the facts at the call prove the condition. An argument the condition
// names that is a handle has its object's facts just before the call.
module {
  tessera.define private @lib.sqrt(f64) -> f64 attributes {argModes = [unit], pure = true}
  llvm.func @trace(!llvm.ptr) -> f64 attributes {tessera.conditional_guarantees = [{condition = "SPD(A)", constants = {}, names = {A = 0 : i64}, property = "positive", result = 0 : i64}]}

  // CHECK-LABEL: llvm.func @of_spd
  // CHECK: tessera.call @lib.sqrt({{.*}}) <arg_attrs = [{tessera.property = ["positive"]}]>
  llvm.func @of_spd(%A: !llvm.ptr {tessera.assumes = ["SPD"]}) -> f64 {
    %t = llvm.call @trace(%A) : (!llvm.ptr) -> f64
    %r = tessera.call @lib.sqrt(%t) : (f64) -> f64
    llvm.return %r : f64
  }

  // CHECK-LABEL: llvm.func @of_any
  // CHECK: tessera.call @lib.sqrt({{[^)]*}}) :
  llvm.func @of_any(%A: !llvm.ptr) -> f64 {
    %t = llvm.call @trace(%A) : (!llvm.ptr) -> f64
    %r = tessera.call @lib.sqrt(%t) : (f64) -> f64
    llvm.return %r : f64
  }
}

// -----

// A val=inout argument keeps the property in the value written back: the
// condition reads the value passed in.
!vec = !llvm.struct<"Vec", (array<4 x i32>)>
module {
  tessera.define private @lib.sorted_vec(!llvm.ptr {llvm.sret = !vec, tessera.guarantees = ["sorted"]}, i64) attributes {argModes = [unit], pure = true}
  tessera.define private @lib.any_vec(!llvm.ptr {llvm.sret = !vec}, i64) attributes {argModes = [unit], pure = true}
  tessera.define private @lib.normalize(!llvm.ptr) attributes {argModes = [{dir = #tessera.dir<inout>, type = !vec}], pure = true, tessera.conditional_guarantees = [{arg = 0 : i64, condition = "sorted(v)", constants = {}, names = {v = 0 : i64}, property = "sorted"}]}
  tessera.define private @lib.use(!llvm.ptr) -> i32 attributes {argModes = [{dir = #tessera.dir<in>, type = !vec}], pure = true}

  // CHECK-LABEL: llvm.func @inout
  // CHECK: %[[T:.*]] = tessera.call @lib.normalize
  // CHECK: tessera.call @lib.use(%[[T]]) <arg_attrs = [{tessera.property = ["sorted"]}]>
  // CHECK: %[[U:.*]] = tessera.call @lib.normalize
  // CHECK: tessera.call @lib.use(%[[U]]) :
  llvm.func @inout(%n: i64) -> i32 {
    %s = tessera.call @lib.sorted_vec(%n) : (i64) -> !vec
    %t = tessera.call @lib.normalize(%s) : (!vec) -> !vec
    %r1 = tessera.call @lib.use(%t) : (!vec) -> i32
    %a = tessera.call @lib.any_vec(%n) : (i64) -> !vec
    %u = tessera.call @lib.normalize(%a) : (!vec) -> !vec
    %r2 = tessera.call @lib.use(%u) : (!vec) -> i32
    %r = llvm.add %r1, %r2 : i32
    llvm.return %r : i32
  }
}
