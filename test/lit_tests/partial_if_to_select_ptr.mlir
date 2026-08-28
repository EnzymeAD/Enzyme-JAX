// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm.func(canonicalize-loops))" | FileCheck %s

// Pointer arithmetic is as safe to speculate as integer arithmetic, so a
// pure pointer-yielding branch becomes a select without speculate_if.
llvm.func @ptr_if_to_select(%c: i1, %base: !llvm.ptr, %i: i64) -> !llvm.ptr {
  %r = scf.if %c -> (!llvm.ptr) {
    scf.yield %base : !llvm.ptr
  } else {
    %g = llvm.getelementptr inbounds %base[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    scf.yield %g : !llvm.ptr
  }
  llvm.return %r : !llvm.ptr
}

// CHECK-LABEL: llvm.func @ptr_if_to_select(
// CHECK-SAME: %[[C:[a-z0-9]+]]: i1, %[[BASE:[a-z0-9]+]]: !llvm.ptr, %[[I:[a-z0-9]+]]: i64
// CHECK-NOT: scf.if
// CHECK: %[[G:.+]] = llvm.getelementptr inbounds %[[BASE]][%[[I]]]
// CHECK: %[[S:.+]] = arith.select %[[C]], %[[BASE]], %[[G]] : !llvm.ptr
// CHECK: llvm.return %[[S]] : !llvm.ptr
