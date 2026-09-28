// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-propagate-properties -tessera-apply-pdl -split-input-file | FileCheck %s

// When the condition is already known to hold, the rewrite is applied outright
// and no check is emitted. This is a saving over the guarded form, never a
// precondition for it: an unprovable condition still works, it just costs a
// check at run time.
//
// Matrices are passed as llvm-to-tessera leaves them: a `const Mat3 &`
// parameter marked val=in takes the struct value, and a `Mat3` returned by
// value is an sret whose call produces the struct.

// A producing function declares what it returns. Written once on the
// declaration, it holds at every call site.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.build_cov(!llvm.ptr {llvm.sret = !mat, tessera.property = ["symmetric"]}, i64) attributes {argModes = [unit], pure = true}
  tessera.define private @lib.foo(!llvm.ptr) -> f32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}
  tessera.define private @lib.symmetric_foo(!llvm.ptr) -> f32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}

  tessera.optimizations {
    tessera.optimization "if symmetric(x), lib.foo(x) -> lib.symmetric_foo(x)"
  }

  // CHECK-LABEL: llvm.func @guaranteed_result
  llvm.func @guaranteed_result(%n: i64) -> f32 {
    // CHECK: %[[M:.*]] = tessera.call @lib.build_cov
    // CHECK-NEXT: %[[R:.*]] = tessera.call @lib.symmetric_foo(%[[M]])
    // CHECK-NEXT: llvm.return %[[R]]
    // CHECK-NOT: tessera.guard
    %0 = tessera.call @lib.build_cov(%n) : (i64) -> !mat
    %1 = tessera.call @lib.foo(%0) : (!mat) -> f32
    llvm.return %1 : f32
  }
}

// -----

// The same matrix without the guarantee gets a guard, which is what makes the
// case above a genuine elision rather than the only path.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.build_any(!llvm.ptr {llvm.sret = !mat}, i64) attributes {argModes = [unit], pure = true}
  tessera.define private @lib.foo(!llvm.ptr) -> f32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}
  tessera.define private @lib.symmetric_foo(!llvm.ptr) -> f32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}

  tessera.optimizations {
    tessera.optimization "if symmetric(x), lib.foo(x) -> lib.symmetric_foo(x)"
  }

  // CHECK-LABEL: llvm.func @no_guarantee
  // CHECK: tessera.guard "symmetric(x)"
  llvm.func @no_guarantee(%n: i64) -> f32 {
    %0 = tessera.call @lib.build_any(%n) : (i64) -> !mat
    %1 = tessera.call @lib.foo(%0) : (!mat) -> f32
    llvm.return %1 : f32
  }
}

// -----

// A property can also be stated on the operation that produced the value.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.foo(!llvm.ptr) -> f32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}
  tessera.define private @lib.diagonal_foo(!llvm.ptr) -> f32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}

  tessera.optimizations {
    tessera.optimization "if diagonal(x), lib.foo(x) -> lib.diagonal_foo(x)"
  }

  // CHECK-LABEL: llvm.func @property_on_value
  // CHECK: tessera.call @lib.diagonal_foo
  // CHECK-NOT: tessera.guard
  llvm.func @property_on_value() -> f32 {
    %0 = llvm.mlir.zero {tessera.property = ["diagonal"]} : !mat
    %1 = tessera.call @lib.foo(%0) : (!mat) -> f32
    llvm.return %1 : f32
  }
}

// -----

// A comparison against a constant folds at compile time, so a rule guarded on
// a size that is already known costs nothing.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.qux(!llvm.ptr, i64) -> f32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, unit], pure = true}
  tessera.define private @lib.tiled_qux(!llvm.ptr, i64) -> f32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, unit], pure = true}

  tessera.optimizations {
    tessera.optimization "if n > 64, lib.qux(x, n) -> lib.tiled_qux(x, n)"
  }

  // CHECK-LABEL: llvm.func @constant_folds_true
  // CHECK: tessera.call @lib.tiled_qux
  // CHECK-NOT: tessera.guard
  // CHECK-NOT: llvm.icmp
  llvm.func @constant_folds_true(%x: !mat) -> f32 {
    %n = llvm.mlir.constant(128 : i64) : i64
    %0 = tessera.call @lib.qux(%x, %n) : (!mat, i64) -> f32
    llvm.return %0 : f32
  }
}

// -----

// The same rule with a non-constant size still guards.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.qux(!llvm.ptr, i64) -> f32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, unit], pure = true}
  tessera.define private @lib.tiled_qux(!llvm.ptr, i64) -> f32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, unit], pure = true}

  tessera.optimizations {
    tessera.optimization "if n > 64, lib.qux(x, n) -> lib.tiled_qux(x, n)"
  }

  // CHECK-LABEL: llvm.func @dynamic_size_guards
  // CHECK: tessera.guard "n > 64"
  llvm.func @dynamic_size_guards(%x: !mat, %n: i64) -> f32 {
    %0 = tessera.call @lib.qux(%x, %n) : (!mat, i64) -> f32
    llvm.return %0 : f32
  }
}

// -----

// Conjunction: both sides provable, so the whole condition is.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.build_cov(!llvm.ptr {llvm.sret = !mat, tessera.property = ["symmetric"]}, i64) attributes {argModes = [unit], pure = true}
  tessera.define private @lib.qux(!llvm.ptr, i64) -> f32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, unit], pure = true}
  tessera.define private @lib.special_qux(!llvm.ptr, i64) -> f32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, unit], pure = true}

  tessera.optimizations {
    tessera.optimization "if symmetric(x) && n > 64, lib.qux(x, n) -> lib.special_qux(x, n)"
  }

  // CHECK-LABEL: llvm.func @conjunction_both_proven
  // CHECK: tessera.call @lib.special_qux
  // CHECK-NOT: tessera.guard
  llvm.func @conjunction_both_proven(%k: i64) -> f32 {
    %n = llvm.mlir.constant(128 : i64) : i64
    %m = tessera.call @lib.build_cov(%k) : (i64) -> !mat
    %0 = tessera.call @lib.qux(%m, %n) : (!mat, i64) -> f32
    llvm.return %0 : f32
  }
}
