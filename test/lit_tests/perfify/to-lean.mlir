// RUN: enzymexlamlir-opt %s -split-input-file -perfify-to-lean -verify-diagnostics -o /dev/null | FileCheck %s
// RUN: enzymexlamlir-opt %s -split-input-file -perfify-to-lean=imports=Stub,Other -verify-diagnostics -o /dev/null | FileCheck %s --check-prefix=IMPORTS

// satres and perfify.loop_trip are set by hand throughout, so the emitted Lean
// does not depend on what Z3 can prove.

// Each perfify.cost becomes a named Nat cost and a TimeM tick that charges it.
// A sym_cost becomes a Nat parameter of both definitions.
module {
  perfify.define_symbol @C : i64
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.addi" {
      %c = perfify.sym_cost @C : i64
      %c2 = arith.constant 2 : i64
      %sq = arith.muli %c, %c : i64
      %ad = arith.addi %sq, %c2 : i64
      perfify.yield %ad : i64
    }
  }
}

// Without imports=, the file imports Mathlib and Cslib's TimeM.
// CHECK:      import Mathlib
// CHECK-NEXT: import Cslib.Algorithms.Lean.TimeM
// CHECK-EMPTY:
// CHECK-NEXT: -- Unproven conditions:
// CHECK:      def «arith.muli» : Nat := 2
// CHECK-NEXT: def «arith.muli».tick : TimeM Nat PUnit := TimeM.tick «arith.muli»
// CHECK-NEXT: def «arith.addi» (sym_C : Nat) : Nat := ((sym_C * sym_C) + 2)
// CHECK-NEXT: def «arith.addi».tick (sym_C : Nat) : TimeM Nat PUnit := TimeM.tick («arith.addi» sym_C)

// IMPORTS-NOT:  import Mathlib
// IMPORTS:      import Stub
// IMPORTS-NEXT: import Other
// IMPORTS-EMPTY:
// IMPORTS-NEXT: -- Unproven conditions:

// -----

// Two perfify.conditions on @named give theorems named apart by the index of the
// condition, so that each theorem maps back to its condition. satres is set by
// hand so the test does not depend on what Z3 can prove.
module {
  func.func @named(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "func.return" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @named true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 9 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
    perfify.conditions @named true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 1 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp ge, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
  }
}

// CHECK-LABEL: -- Unproven conditions: @named{{$}}
// CHECK: def «named» : TimeM Nat PUnit := do
// CHECK: def «named».pre_0 (pre : Nat) : Prop := pre = 0
// CHECK-NEXT: theorem «named».post_0 (pre : Nat) (hpre : «named».pre_0 pre) : pre + «named».time ≤ 9 := by
// CHECK: def «named».pre_1 (pre : Nat) : Prop := pre = 0
// CHECK-NEXT: theorem «named».post_1 (pre : Nat) (hpre : «named».pre_1 pre) : pre + «named».time ≥ 1 := by
// CHECK-NOT: theorem

// -----

// A pre hypothesis is a premise of each post theorem, not a theorem of its
// own. A parameter that both use, sym_N here, is bound once.
module {
  perfify.define_symbol @N : i64
  func.func @shared(%a0: i64) -> i64 {
    return %a0 : i64
  }
  perfify.assumptions {
    perfify.cost "func.return" {
      %c = arith.constant 1 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @shared true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %n = perfify.sym_cost @N : i64
        perfify.yield %n : i64
      }
      %cmp = perfify.cmp le, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %n = perfify.sym_cost @N : i64
        %c = arith.constant 1 : i64
        %s = arith.addi %n, %c : i64
        perfify.yield %s : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
  }
}

// CHECK-LABEL: -- Unproven conditions: @shared{{$}}
// CHECK-NOT:   theorem «shared».pre
// CHECK:       def «shared».pre_0 (pre : Nat) (sym_N : Nat) : Prop := pre ≤ sym_N
// CHECK-NEXT:  theorem «shared».post_0 (pre : Nat) (sym_N : Nat) (hpre : «shared».pre_0 pre sym_N) : pre + «shared».time ≤ (sym_N + 1) := by
// CHECK-NOT:   theorem

// -----

// The Lean definition charges each op the cost the analysis gives it: its own
// perfify.cost, else its dialect's "<dialect>.*", else "*".
module {
  func.func @fallback(%a0: i64) -> i64 {
    %m = arith.muli %a0, %a0 : i64
    %s = arith.addi %m, %a0 : i64
    return %s : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 3 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.*" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.cost "*" {
      %c = arith.constant 1 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @fallback true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 10 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
  }
}

// CHECK-LABEL: def «fallback» : TimeM Nat PUnit := do
// CHECK-NEXT:    «arith.muli».tick
// CHECK-NEXT:    «arith.*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:  def «fallback».pre_0 (pre : Nat) : Prop := pre = 0
// CHECK-NEXT:  theorem «fallback».post_0 (pre : Nat) (hpre : «fallback».pre_0 pre) : pre + «fallback».time ≤ 10 := by

// -----

// An op with no cost leaves the analysis unproven, so it must not be free in
// Lean either: the definition is sorry, and a proof about it fails the axiom
// check.
module {
  func.func @uncosted(%a0: i64) -> i64 { // expected-warning {{function body has no Lean form, emitting sorry}}
    %m = arith.muli %a0, %a0 : i64
    return %m : i64
  }
  perfify.assumptions {
    perfify.cost "arith.muli" {
      %c = arith.constant 3 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @uncosted true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp ge, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 10 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
  }
}

// CHECK: def «uncosted» : TimeM Nat PUnit := sorry

// -----

// A call has no Lean form yet, even under "*".
module {
  func.func private @callee(i64) -> i64
  func.func @caller(%a0: i64) -> i64 { // expected-warning {{function body has no Lean form, emitting sorry}}
    %r = func.call @callee(%a0) : (i64) -> i64
    return %r : i64
  }
  perfify.assumptions {
    perfify.cost "*" {
      %c = arith.constant 1 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @caller true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp ge, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 10 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
  }
}

// CHECK: def «caller» : TimeM Nat PUnit := sorry

// -----

// A loop charges its own cost once, then its body once per iteration, as
// (List.range trips).forM. A static trip count is a literal.
module {
  func.func @static(%a0: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %a0) -> (i64) {
      %s = arith.addi %acc, %acc : i64
      scf.yield %s : i64
    }
    return %r : i64
  }
  perfify.assumptions {
    perfify.cost "*" {
      %c = arith.constant 1 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @static true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp ge, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 1 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
  }
}

// CHECK-LABEL: def «static» : TimeM Nat PUnit := do
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    (List.range 4).forM fun i0 => do
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:  def «static».pre_0 (pre : Nat) : Prop := pre ≥ 0
// CHECK-NEXT:  theorem «static».post_0

// -----

// A dynamic trip count is the Nat parameter for the loop's perfify.loop_trip
// symbol, which the analysis would define.
module {
  perfify.define_symbol @T : i64
  func.func @dynamic(%a0: i64, %n: index) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a0) -> (i64) {
      %s = arith.addi %acc, %acc : i64
      scf.yield %s : i64
    } {perfify.loop_trip = @T}
    return %r : i64
  }
  perfify.assumptions {
    perfify.cost "*" {
      %c = arith.constant 1 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @dynamic true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 1 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
  }
}

// CHECK-LABEL: def «dynamic» (sym_T : Nat) : TimeM Nat PUnit := do
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    (List.range sym_T).forM fun i0 => do
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:  def «dynamic».pre_0 (pre : Nat) : Prop := pre = 0
// CHECK-NEXT:  theorem «dynamic».post_0 (pre : Nat) (sym_T : Nat) (hpre : «dynamic».pre_0 pre) : pre + («dynamic» sym_T).time ≤ 1 := by

// -----

// A nested loop is indented one level further inside its parent's body.
module {
  perfify.define_symbol @N : i64
  func.func @nested(%a0: i64, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %n step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        %s = arith.addi %a0, %a0 : i64
      }
    } {perfify.loop_trip = @N}
    return
  }
  perfify.assumptions {
    perfify.cost "*" {
      %c = arith.constant 1 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @nested true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp ge, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 1 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
  }
}

// CHECK-LABEL: def «nested» (sym_N : Nat) : TimeM Nat PUnit := do
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    (List.range sym_N).forM fun i0 => do
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:      (List.range 2).forM fun i1 => do
// CHECK-NEXT:        «*».tick
// CHECK-NEXT:        «*».tick
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:  def «nested».pre_0 (pre : Nat) : Prop := pre ≥ 0
// CHECK-NEXT:  theorem «nested».post_0

// -----

// Each branch is chosen by its own Bool parameter, which takes the indices of
// the loops around it, as the analysis lets each iteration take either arm.
// An if without an else has an empty else arm.
module {
  perfify.define_symbol @T : i64
  func.func @branches(%a0: i64, %k: i1, %n: index) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = scf.if %k -> (i64) {
      %m = arith.muli %a0, %a0 : i64
      scf.yield %m : i64
    } else {
      scf.yield %a0 : i64
    }
    scf.for %i = %c0 to %n step %c1 {
      scf.if %k {
        %s = arith.addi %a0, %a0 : i64
      }
    } {perfify.loop_trip = @T}
    return %r : i64
  }
  perfify.assumptions {
    perfify.cost "*" {
      %c = arith.constant 1 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @branches true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 1 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
  }
}

// CHECK-LABEL: def «branches» (branch0 : List Nat -> Bool) (sym_T : Nat) (branch1 : List Nat -> Bool) : TimeM Nat PUnit := do
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    if branch0 [] then do
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:    else do
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    (List.range sym_T).forM fun i0 => do
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:      if branch1 [i0] then do
// CHECK-NEXT:        «*».tick
// CHECK-NEXT:        «*».tick
// CHECK-NEXT:      else do
// CHECK-NEXT:        pure ()
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:  def «branches».pre_0 (pre : Nat) : Prop := pre = 0
// CHECK-NEXT:  theorem «branches».post_0 (pre : Nat) (branch0 : List Nat -> Bool) (sym_T : Nat) (branch1 : List Nat -> Bool) (hpre : «branches».pre_0 pre) : pre + («branches» branch0 sym_T branch1).time ≤ 1 := by

// -----

// A call costs what its perfify.func_bound gives, as a premise on a cost
// parameter that takes the indices of the loops around the call. The call op
// itself is not charged. A call with no func_bound has no Lean form.
module {
  perfify.define_symbol @T : i64
  func.func private @ext(i64) -> i64
  func.func private @unbounded(i64) -> i64
  func.func @contracted(%a0: i64, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %r = func.call @ext(%a0) : (i64) -> i64
    } {perfify.loop_trip = @T}
    return
  }
  func.func @uncontracted(%a0: i64) -> i64 { // expected-warning {{function body has no Lean form, emitting sorry}}
    %r = func.call @unbounded(%a0) : (i64) -> i64
    return %r : i64
  }
  perfify.func_bound @ext eq {
    perfify.estimated_cost @func_cost {
      %c = arith.constant 10 : i64
      perfify.yield %c : i64
    }
  }
  perfify.assumptions {
    perfify.cost "*" {
      %c = arith.constant 1 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @contracted true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 1 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
    perfify.conditions @uncontracted true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp ge, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 1 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
  }
}

// CHECK-LABEL: def «contracted» (sym_T : Nat) (call0 : List Nat -> Nat) : TimeM Nat PUnit := do
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    (List.range sym_T).forM fun i0 => do
// CHECK-NEXT:      TimeM.tick (call0 [i0])
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:  def «contracted».pre_0 (pre : Nat) : Prop := pre = 0
// CHECK-NEXT:  theorem «contracted».post_0 (pre : Nat) (call0 : List Nat -> Nat) (sym_T : Nat) (hpre : «contracted».pre_0 pre) (hbody_0 : ∀ l, (call0 l) = 10) : pre + («contracted» sym_T call0).time ≤ 1 := by
// CHECK:       def «uncontracted» : TimeM Nat PUnit := sorry

// -----

// The entry block of a CFG runs once, and each other block may or may not run,
// as the analysis counts each such block independently. A block in a cycle
// may run any number of times, so a CFG with a cycle has no Lean form.
module {
  func.func @diamond(%a0: i64, %k: i1) -> i64 {
    cf.cond_br %k, ^bb1, ^bb2
  ^bb1:
    %x = arith.muli %a0, %a0 : i64
    cf.br ^bb3(%x : i64)
  ^bb2:
    cf.br ^bb3(%a0 : i64)
  ^bb3(%r: i64):
    return %r : i64
  }
  func.func @cycle(%a0: i64) -> i64 { // expected-warning {{function body has no Lean form, emitting sorry}}
    cf.br ^bb1
  ^bb1:
    cf.br ^bb1
  }
  perfify.assumptions {
    perfify.cost "*" {
      %c = arith.constant 1 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @diamond true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp ge, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 1 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
    perfify.conditions @cycle true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp ge, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 1 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
  }
}

// CHECK-LABEL: def «diamond» (branch0 : List Nat -> Bool) (branch1 : List Nat -> Bool) (branch2 : List Nat -> Bool) : TimeM Nat PUnit := do
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    if branch0 [] then do
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:    else do
// CHECK-NEXT:      pure ()
// CHECK-NEXT:    if branch1 [] then do
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:    else do
// CHECK-NEXT:      pure ()
// CHECK-NEXT:    if branch2 [] then do
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:    else do
// CHECK-NEXT:      pure ()
// CHECK-NEXT:  def «diamond».pre_0 (pre : Nat) : Prop := pre ≥ 0
// CHECK-NEXT:  theorem «diamond».post_0
// CHECK:       def «cycle» : TimeM Nat PUnit := sorry

// -----

// A dynamic loop nested in another loop gets a trip count for each iteration
// of the loops around it, so the iterations of the outer loop can run the
// inner one different numbers of times. Its hypothesis holds for each of them.
module {
  perfify.define_symbol @R : i64
  perfify.define_symbol @U : i64
  func.func @rows(%a0: i64, %n: index, %m: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      scf.for %j = %c0 to %m step %c1 {
        %b = perfify.arg 0
        perfify.estimated_cost @trip_cost {
          %c27 = arith.constant 27 : i64
          perfify.yield %c27 : i64
        }
        %cmp = perfify.cmp le, %b, @trip_cost
        %h = perfify.hypothesis %cmp
        %s = arith.addi %a0, %a0 : i64
      } {perfify.loop_trip = @U}
    } {perfify.loop_trip = @R}
    return
  }
  perfify.assumptions {
    perfify.cost "*" {
      %c = arith.constant 1 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @rows true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 1 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
  }
}

// CHECK-LABEL: def «rows» (sym_R : Nat) (iter_U : List Nat -> Nat) : TimeM Nat PUnit := do
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:    (List.range sym_R).forM fun i0 => do
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:      (List.range (iter_U [i0])).forM fun i1 => do
// CHECK-NEXT:        «*».tick
// CHECK-NEXT:        «*».tick
// CHECK-NEXT:      «*».tick
// CHECK-NEXT:    «*».tick
// CHECK-NEXT:  def «rows».pre_0 (pre : Nat) : Prop := pre = 0
// CHECK-NEXT:  theorem «rows».post_0 (pre : Nat) (iter_U : List Nat -> Nat) (sym_R : Nat) (hpre : «rows».pre_0 pre) (hbody_0 : ∀ l, (iter_U l) ≤ 27) : pre + («rows» sym_R iter_U).time ≤ 1 := by

// -----

// A condition with a counterexample region still gets a theorem, since added
// premises can give Lean what Z3 lacked. The region is shown above it.
module {
  func.func @disproved(%a0: i64) -> i64 {
    return %a0 : i64
  }
  func.func @open(%a0: i64) -> i64 {
    return %a0 : i64
  }
  perfify.assumptions {
    perfify.cost "*" {
      %c = arith.constant 1 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @disproved true invalid_region = "true" pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp ge, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 1 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
    perfify.conditions @open true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp ge, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %c = arith.constant 1 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
  }
}

// CHECK-LABEL: -- Unproven conditions: @disproved @open{{$}}
// CHECK:       -- Z3 counterexample: true
// CHECK-NEXT:  theorem «disproved».post_0
// CHECK-NOT:   -- Z3 counterexample
// CHECK:       theorem «open».post_0

// -----

// The trip count hypothesis of a dynamic loop is a premise of each post
// theorem, as it is an assumption of the analysis. The cost is
// 3 + 2 * trips + 1, so the bound 2 * N + 4 follows from trips <= N.
module {
  perfify.define_symbol @N : i64
  perfify.define_symbol @T : i64
  func.func @bounded(%a0: i64, %n: index) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a0) -> (i64) {
      %b = perfify.arg 0
      perfify.estimated_cost @trip_cost {
        %t = perfify.sym_cost @N : i64
        perfify.yield %t : i64
      }
      %cmp = perfify.cmp le, %b, @trip_cost
      %h = perfify.hypothesis %cmp
      %s = arith.addi %acc, %acc : i64
      scf.yield %s : i64
    } {perfify.loop_trip = @T}
    return %r : i64
  }
  perfify.assumptions {
    perfify.cost "*" {
      %c = arith.constant 1 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @bounded true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %n = perfify.sym_cost @N : i64
        %c2 = arith.constant 2 : i64
        %c4 = arith.constant 4 : i64
        %m = arith.muli %c2, %n : i64
        %s = arith.addi %m, %c4 : i64
        perfify.yield %s : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
  }
}

// CHECK: def «bounded» (sym_T : Nat) : TimeM Nat PUnit := do
// CHECK: def «bounded».pre_0 (pre : Nat) : Prop := pre = 0
// CHECK-NEXT: theorem «bounded».post_0 (pre : Nat) (sym_T : Nat) (sym_N : Nat) (hpre : «bounded».pre_0 pre) (hbody_0 : sym_T ≤ sym_N) : pre + («bounded» sym_T).time ≤ ((2 * sym_N) + 4) := by

// -----

// A self-recursive function runs its body once per activation, so the body is
// wrapped in a loop over a free activation count. Choices inside are indexed
// by the activation, and the self-call costs nothing, as the later activations
// pay for it.
module {
  perfify.define_symbol @N : i64
  func.func @countdown(%n: i64) -> i64 {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %z = arith.cmpi eq, %n, %c0 : i64
    %r = scf.if %z -> i64 {
      scf.yield %c0 : i64
    } else {
      %m = arith.subi %n, %c1 : i64
      %w = arith.muli %m, %m : i64
      %x = func.call @countdown(%m) : (i64) -> i64
      scf.yield %x : i64
    }
    return %r : i64
  }
  perfify.assumptions {
    perfify.cost "*" {
      %c = arith.constant 0 : i64
      perfify.yield %c : i64
    }
    perfify.cost "arith.muli" {
      %c = arith.constant 2 : i64
      perfify.yield %c : i64
    }
    perfify.conditions @countdown true pre {
      %b0 = perfify.arg 0
      perfify.estimated_cost @pre_cost {
        %c = arith.constant 0 : i64
        perfify.yield %c : i64
      }
      %cmp = perfify.cmp eq, %b0, @pre_cost
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h {satres = true}
    } post {
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @post_cost {
        %s = perfify.sym_cost @N : i64
        %c2 = arith.constant 2 : i64
        %m = arith.muli %c2, %s : i64
        %t = arith.addi %m, %c2 : i64
        perfify.yield %t : i64
      }
      %cmp = perfify.cmp le, %cost, @post_cost
      %h = perfify.hypothesis %cmp post
      perfify.assume %h {satres = false}
    }
  }
}

// CHECK-LABEL: def «countdown» (activations : Nat) (branch0 : List Nat -> Bool) : TimeM Nat PUnit := do
// CHECK-NEXT:    (List.range activations).forM fun i0 => do
// CHECK-NOT:     call
// CHECK:         if branch0 [i0] then do
// CHECK-NOT:     call
// CHECK:       def «countdown».pre_0 (pre : Nat) : Prop := pre = 0
// CHECK-NEXT:  theorem «countdown».post_0 (pre : Nat) (activations : Nat) (branch0 : List Nat -> Bool) (sym_N : Nat) (hpre : «countdown».pre_0 pre) : pre + («countdown» activations branch0).time ≤ ((2 * sym_N) + 2) := by
