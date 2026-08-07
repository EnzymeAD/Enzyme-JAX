// RUN: enzymexlamlir-opt %s --enzyme-lift-cf-to-scf | FileCheck %s --check-prefix=LIFT
// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s --check-prefix=ACCESS

// An unwind edge has no reading in scf, and none of the raising passes may
// pretend it is not there: lowering invoke to call compiled Catch2's
// exception-driven test runner into a binary that ran one test case of 103
// and exited clean. A function that throws or catches stays in cf form --
// correct, just never raised -- and everything around it raises as usual.

module {
  llvm.func @__gxx_personality_v0(...) -> i32
  llvm.func @maythrow() -> i32
  llvm.func @sink(f64)
  llvm.func @_Znwm(i64) -> !llvm.ptr

  llvm.func @keeps_eh(%c: i1) -> i32 attributes {personality = @__gxx_personality_v0} {
    %0 = llvm.invoke @maythrow() to ^ok unwind ^lp : () -> i32
  ^ok:
    llvm.return %0 : i32
  ^lp:
    %1 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.resume %1 : !llvm.struct<(ptr, i32)>
  }

  // The result of an invoke only exists on its normal edge: what the raising
  // computes from it -- the memref view of the allocation here -- must be
  // materialized in the successor, not after the terminator.
  llvm.func @invoke_base(%n: i64) attributes {personality = @__gxx_personality_v0} {
    %c32 = llvm.mlir.constant(32 : i64) : i64
    %p = llvm.invoke @_Znwm(%c32) to ^ok unwind ^lp : (i64) -> !llvm.ptr
  ^ok:
    %q = llvm.getelementptr inbounds %p[%n] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %v = llvm.load %q {alignment = 8 : i64} : !llvm.ptr -> f64
    llvm.call @sink(%v) : (f64) -> ()
    llvm.return
  ^lp:
    %1 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.resume %1 : !llvm.struct<(ptr, i32)>
  }
}

// LIFT-LABEL: llvm.func @keeps_eh
// LIFT:         llvm.invoke @maythrow
// LIFT:         llvm.landingpad
// LIFT:         llvm.resume

// ACCESS-LABEL: llvm.func @invoke_base
// ACCESS:         %[[P:.+]] = llvm.invoke @_Znwm{{.*}} to ^bb1 unwind ^bb2
// ACCESS:       ^bb1:
// ACCESS-NEXT:    %[[M:.+]] = "enzymexla.pointer2memref"(%[[P]])
// ACCESS-NEXT:    %[[V:.+]] = affine.load %[[M]][symbol(%{{.+}})] {alignment = 8 : i64
// ACCESS:         llvm.landingpad
