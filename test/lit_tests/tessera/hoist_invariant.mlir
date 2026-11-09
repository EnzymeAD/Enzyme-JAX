// RUN: enzymexlamlir-opt %s -split-input-file -tessera-hoist-invariant | FileCheck %s

// A call to a pure tessera op moves out of a loop when its operands are defined
// outside it and nothing in the loop may change the objects it reads.

module {
  tessera.define private @lib.norm(!llvm.ptr) -> f64 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.update(!llvm.ptr) -> f64 attributes {argModes = [unit], pure = false}
  tessera.define private @lib.twice(f64) -> f64 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.make(!llvm.ptr {llvm.nocapture}) -> !llvm.ptr attributes {argModes = [unit], pure = true}
  tessera.define private @lib.sum(!llvm.ptr) -> f64 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.release(!llvm.ptr) attributes {argModes = [unit], pure = false}
  llvm.func @use(f64)
  llvm.func @scale(!llvm.ptr)
  llvm.func @attach(!llvm.ptr, !llvm.ptr {tessera.unchanged})
  llvm.func @touch(!llvm.ptr)

  // CHECK-LABEL: llvm.func @invariant
  // CHECK: tessera.call @lib.norm
  // CHECK-NEXT: scf.for
  // CHECK-NOT: tessera.call @lib.norm
  // CHECK: llvm.return
  llvm.func @invariant(%a: !llvm.ptr {llvm.noalias}, %n: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ni = arith.index_cast %n : i64 to index
    scf.for %i = %c0 to %ni step %c1 {
      %v = tessera.call @lib.norm(%a) : (!llvm.ptr) -> f64
      llvm.call @use(%v) : (f64) -> ()
    }
    llvm.return
  }

  // The loop changes the object.
  // CHECK-LABEL: llvm.func @changed_in_loop
  // CHECK: scf.for
  // CHECK-NEXT: tessera.call @lib.norm
  llvm.func @changed_in_loop(%a: !llvm.ptr {llvm.noalias}, %n: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ni = arith.index_cast %n : i64 to index
    scf.for %i = %c0 to %ni step %c1 {
      %v = tessera.call @lib.norm(%a) : (!llvm.ptr) -> f64
      llvm.call @use(%v) : (f64) -> ()
      llvm.call @scale(%a) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  // ctx may hold the object, and the loop gives ctx to a call that may change
  // what it holds.
  // CHECK-LABEL: llvm.func @holder_in_loop
  // CHECK: scf.for
  // CHECK-NEXT: tessera.call @lib.norm
  llvm.func @holder_in_loop(%a: !llvm.ptr {llvm.noalias}, %ctx: !llvm.ptr, %n: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ni = arith.index_cast %n : i64 to index
    llvm.call @attach(%ctx, %a) : (!llvm.ptr, !llvm.ptr) -> ()
    scf.for %i = %c0 to %ni step %c1 {
      %v = tessera.call @lib.norm(%a) : (!llvm.ptr) -> f64
      llvm.call @use(%v) : (f64) -> ()
      llvm.call @touch(%ctx) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  // Only what a tessera call needs moves: other pure ops are left to the
  // pipeline's own loop-invariant code motion.
  // CHECK-LABEL: llvm.func @only_operands
  // CHECK: arith.addf
  // CHECK-NEXT: tessera.call @lib.twice
  // CHECK-NEXT: scf.for
  // CHECK-NEXT: arith.mulf
  llvm.func @only_operands(%x: f64, %n: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ni = arith.index_cast %n : i64 to index
    scf.for %i = %c0 to %ni step %c1 {
      %y = arith.mulf %x, %x : f64
      llvm.call @use(%y) : (f64) -> ()
      %z = arith.addf %x, %x : f64
      %v = tessera.call @lib.twice(%z) : (f64) -> f64
      llvm.call @use(%v) : (f64) -> ()
    }
    llvm.return
  }

  // What the call returns is only read in the loop, so it moves, and so does
  // the pure op that reads it.
  // CHECK-LABEL: llvm.func @result_read
  // CHECK: tessera.call @lib.make
  // CHECK-NEXT: tessera.call @lib.sum
  // CHECK-NEXT: scf.for
  llvm.func @result_read(%a: !llvm.ptr {llvm.noalias}, %n: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ni = arith.index_cast %n : i64 to index
    scf.for %i = %c0 to %ni step %c1 {
      %t = tessera.call @lib.make(%a) : (!llvm.ptr) -> !llvm.ptr
      %v = tessera.call @lib.sum(%t) : (!llvm.ptr) -> f64
      llvm.call @use(%v) : (f64) -> ()
    }
    llvm.return
  }

  // What the call returns is released each step, so moving the call would use
  // it after the first release.
  // CHECK-LABEL: llvm.func @result_released
  // CHECK: scf.for
  // CHECK-NEXT: tessera.call @lib.make
  llvm.func @result_released(%a: !llvm.ptr {llvm.noalias}, %n: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ni = arith.index_cast %n : i64 to index
    scf.for %i = %c0 to %ni step %c1 {
      %t = tessera.call @lib.make(%a) : (!llvm.ptr) -> !llvm.ptr
      %v = tessera.call @lib.sum(%t) : (!llvm.ptr) -> f64
      llvm.call @use(%v) : (f64) -> ()
      tessera.call @lib.release(%t) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  // The argument is computed in the loop.
  // CHECK-LABEL: llvm.func @operand_in_loop
  // CHECK: scf.for
  // CHECK: tessera.call @lib.norm
  llvm.func @operand_in_loop(%slots: !llvm.ptr, %n: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ni = arith.index_cast %n : i64 to index
    scf.for %i = %c0 to %ni step %c1 {
      %j = arith.index_cast %i : index to i64
      %p = llvm.getelementptr %slots[%j] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
      %a = llvm.load %p : !llvm.ptr -> !llvm.ptr
      %v = tessera.call @lib.norm(%a) : (!llvm.ptr) -> f64
      llvm.call @use(%v) : (f64) -> ()
    }
    llvm.return
  }

  // Not pure: it may change memory.
  // CHECK-LABEL: llvm.func @not_pure
  // CHECK: scf.for
  // CHECK-NEXT: tessera.call @lib.update
  llvm.func @not_pure(%a: !llvm.ptr {llvm.noalias}, %n: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ni = arith.index_cast %n : i64 to index
    scf.for %i = %c0 to %ni step %c1 {
      %v = tessera.call @lib.update(%a) : (!llvm.ptr) -> f64
      llvm.call @use(%v) : (f64) -> ()
    }
    llvm.return
  }
}
