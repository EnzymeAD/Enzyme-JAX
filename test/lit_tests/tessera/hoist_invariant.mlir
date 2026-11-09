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

// -----
// Factor reuse: a rule has split a solve on a fresh copy of the matrix into a
// factorization and a solve with it. The copy moves out, under a test that the
// loop runs; then the factorization; then its free goes after the loop.

module {
  tessera.define private @tl.factor(!llvm.ptr {llvm.nocapture}, i64) -> !llvm.ptr attributes {argModes = [unit, unit], pure = true}
  tessera.define private @tl.solve_factored(!llvm.ptr {tessera.unchanged}, !llvm.ptr) attributes {argModes = [unit, unit], pure = false}
  tessera.define private @c.free(!llvm.ptr) attributes {argModes = [unit], pure = false, tessera.original_name = "free"}
  tessera.define private @tl.update(!llvm.ptr, !llvm.ptr) attributes {argModes = [unit, unit], pure = false}
  tessera.define private @tl.release(!llvm.ptr) attributes {argModes = [unit], pure = false}
  llvm.func @scale(!llvm.ptr)

  // CHECK-LABEL: llvm.func @factor_reuse
  // CHECK: %[[RUNS:.*]] = arith.cmpi slt
  // CHECK: scf.if %[[RUNS]] {
  // CHECK-NEXT: "llvm.intr.memcpy"(%[[W:.*]], %[[A:.*]], %{{.*}})
  // CHECK: %[[F:.*]] = tessera.call @tl.factor(%[[W]]
  // CHECK: affine.for
  // CHECK-NEXT: tessera.call @tl.solve_factored(%[[F]]
  // CHECK-NEXT: }
  // CHECK-NEXT: tessera.call @c.free(%[[F]])
  llvm.func @factor_reuse(%a: !llvm.ptr {llvm.noalias}, %w: !llvm.ptr {llvm.noalias}, %u: !llvm.ptr {llvm.noalias}, %bytes: i64, %n: i64, %steps: i64) {
    %m = arith.index_cast %steps : i64 to index
    affine.for %s = 0 to %m {
      "llvm.intr.memcpy"(%w, %a, %bytes) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
      %f = tessera.call @tl.factor(%w, %n) : (!llvm.ptr, i64) -> !llvm.ptr
      tessera.call @tl.solve_factored(%f, %u) : (!llvm.ptr, !llvm.ptr) -> ()
      tessera.call @c.free(%f) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  // The loop changes the source, so every step copies and factors anew.
  // CHECK-LABEL: llvm.func @source_changed
  // CHECK: affine.for
  // CHECK-NEXT: llvm.call @scale
  // CHECK-NEXT: "llvm.intr.memcpy"
  // CHECK-NEXT: tessera.call @tl.factor
  // CHECK-NEXT: tessera.call @tl.solve_factored
  // CHECK-NEXT: tessera.call @c.free
  llvm.func @source_changed(%a: !llvm.ptr {llvm.noalias}, %w: !llvm.ptr {llvm.noalias}, %u: !llvm.ptr {llvm.noalias}, %bytes: i64, %n: i64, %steps: i64) {
    %m = arith.index_cast %steps : i64 to index
    affine.for %s = 0 to %m {
      llvm.call @scale(%a) : (!llvm.ptr) -> ()
      "llvm.intr.memcpy"(%w, %a, %bytes) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
      %f = tessera.call @tl.factor(%w, %n) : (!llvm.ptr, i64) -> !llvm.ptr
      tessera.call @tl.solve_factored(%f, %u) : (!llvm.ptr, !llvm.ptr) -> ()
      tessera.call @c.free(%f) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  // The loop writes the copy after factoring it, so the next step must copy
  // again.
  // CHECK-LABEL: llvm.func @destination_changed
  // CHECK: affine.for
  // CHECK-NEXT: "llvm.intr.memcpy"
  // CHECK-NEXT: tessera.call @tl.factor
  llvm.func @destination_changed(%a: !llvm.ptr {llvm.noalias}, %w: !llvm.ptr {llvm.noalias}, %u: !llvm.ptr {llvm.noalias}, %bytes: i64, %n: i64, %steps: i64) {
    %m = arith.index_cast %steps : i64 to index
    affine.for %s = 0 to %m {
      "llvm.intr.memcpy"(%w, %a, %bytes) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
      %f = tessera.call @tl.factor(%w, %n) : (!llvm.ptr, i64) -> !llvm.ptr
      tessera.call @tl.solve_factored(%f, %w) : (!llvm.ptr, !llvm.ptr) -> ()
      tessera.call @c.free(%f) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  // What the factorization returns is released each step by a call other
  // than free, so moving it would use it after the first release.
  // CHECK-LABEL: llvm.func @released_in_loop
  // CHECK: affine.for
  // CHECK-NEXT: tessera.call @tl.factor
  llvm.func @released_in_loop(%w: !llvm.ptr {llvm.noalias}, %u: !llvm.ptr {llvm.noalias}, %n: i64, %steps: i64) {
    %m = arith.index_cast %steps : i64 to index
    affine.for %s = 0 to %m {
      %f = tessera.call @tl.factor(%w, %n) : (!llvm.ptr, i64) -> !llvm.ptr
      tessera.call @tl.solve_factored(%f, %u) : (!llvm.ptr, !llvm.ptr) -> ()
      tessera.call @tl.release(%f) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  // A call that may change what the factorization returns.
  // CHECK-LABEL: llvm.func @result_changed
  // CHECK: affine.for
  // CHECK-NEXT: tessera.call @tl.factor
  llvm.func @result_changed(%w: !llvm.ptr {llvm.noalias}, %u: !llvm.ptr {llvm.noalias}, %n: i64, %steps: i64) {
    %m = arith.index_cast %steps : i64 to index
    affine.for %s = 0 to %m {
      %f = tessera.call @tl.factor(%w, %n) : (!llvm.ptr, i64) -> !llvm.ptr
      tessera.call @tl.update(%f, %u) : (!llvm.ptr, !llvm.ptr) -> ()
      tessera.call @c.free(%f) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  // What it returns is stored where something after the loop could find it.
  // CHECK-LABEL: llvm.func @result_stored
  // CHECK: affine.for
  // CHECK-NEXT: tessera.call @tl.factor
  llvm.func @result_stored(%w: !llvm.ptr {llvm.noalias}, %slot: !llvm.ptr, %n: i64, %steps: i64) {
    %m = arith.index_cast %steps : i64 to index
    affine.for %s = 0 to %m {
      %f = tessera.call @tl.factor(%w, %n) : (!llvm.ptr, i64) -> !llvm.ptr
      llvm.store %f, %slot : !llvm.ptr, !llvm.ptr
    }
    llvm.return
  }

  // Freed only on some steps: the free can't simply move after the loop.
  // CHECK-LABEL: llvm.func @freed_in_branch
  // CHECK: affine.for
  // CHECK-NEXT: tessera.call @tl.factor
  llvm.func @freed_in_branch(%w: !llvm.ptr {llvm.noalias}, %u: !llvm.ptr {llvm.noalias}, %n: i64, %steps: i64, %c: i1) {
    %m = arith.index_cast %steps : i64 to index
    affine.for %s = 0 to %m {
      %f = tessera.call @tl.factor(%w, %n) : (!llvm.ptr, i64) -> !llvm.ptr
      tessera.call @tl.solve_factored(%f, %u) : (!llvm.ptr, !llvm.ptr) -> ()
      scf.if %c {
        tessera.call @c.free(%f) : (!llvm.ptr) -> ()
      }
    }
    llvm.return
  }

  // A loop that runs no times keeps its copy, which then never runs.
  // CHECK-LABEL: llvm.func @never_runs
  // CHECK-NOT: scf.if
  // CHECK: affine.for
  // CHECK-NEXT: "llvm.intr.memcpy"
  llvm.func @never_runs(%a: !llvm.ptr {llvm.noalias}, %w: !llvm.ptr {llvm.noalias}, %bytes: i64) {
    affine.for %s = 4 to 4 {
      "llvm.intr.memcpy"(%w, %a, %bytes) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    }
    llvm.return
  }
}
