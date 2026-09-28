// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-propagate-properties -tessera-apply-pdl -split-input-file -verify-diagnostics | FileCheck %s

// A property a rule's condition tests comes from declarations alone. What is
// always true sits on the definition, as a `tessera.property` on the output
// guaranteed or the parameter assumed; a function that preserves a property
// carries a `tessera.preserves` rule. tessera-propagate-properties turns these
// into facts on the call arguments, and the condition reads those. Nothing is
// checked at run time.
//
// A name with no runtime check, like `SPD`, holds only by declaration. When it
// cannot be shown the rule does not apply at that call, and says so in a
// remark; no guard is built.
//
// Matrices are what llvm-to-tessera makes of a fixed-size matrix class: a
// `const Mat3 &` parameter marked val=in is passed as the struct value, a
// `Mat3 &` the function writes comes back as a call result, and a `Mat3`
// returned by value is an sret whose call produces the struct.

// A guarantee on a matrix returned by value holds at every call site.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.mass(!llvm.ptr {llvm.sret = !mat, tessera.property = ["SPD"]}, i64) attributes {argModes = [unit], pure = true}
  tessera.define private @lib.inverse(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, {dir = #tessera.dir<out>, type = !mat}], pure = true}
  tessera.define private @lib.inverse_spd(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, {dir = #tessera.dir<out>, type = !mat}], pure = true}

  tessera.optimizations {
    tessera.optimization "if SPD(m), lib.inverse(m) -> lib.inverse_spd(m)"
  }

  // CHECK-LABEL: llvm.func @guaranteed_return
  // CHECK: %[[M:.*]] = tessera.call @lib.mass
  // CHECK-NEXT: %[[R:.*]] = tessera.call @lib.inverse_spd(%[[M]])
  // CHECK-NEXT: llvm.return %[[R]]
  // CHECK-NOT: tessera.guard
  llvm.func @guaranteed_return(%n: i64) -> !mat {
    %m = tessera.call @lib.mass(%n) : (i64) -> !mat
    %r = tessera.call @lib.inverse(%m) : (!mat) -> !mat
    llvm.return %r : !mat
  }
}

// -----

// Carried through functions that preserve it: scaling keeps SPD from its
// matrix argument, and a sum keeps it when both terms have it. Both write
// their result through an out-argument, which is the output a preserves
// declaration means when the function returns nothing.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.mass(!llvm.ptr {llvm.sret = !mat, tessera.property = ["SPD"]}, i64) attributes {argModes = [unit], pure = true}
  tessera.define private @lib.stiffness(!llvm.ptr {llvm.sret = !mat}, i64) attributes {argModes = [unit], pure = true}
  tessera.define private @lib.scale(!llvm.ptr, f64, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, unit, {dir = #tessera.dir<out>, type = !mat}], pure = true, tessera.preserves = [{inputs = [0], property = "SPD"}]}
  tessera.define private @lib.add(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, {dir = #tessera.dir<in>, type = !mat}, {dir = #tessera.dir<out>, type = !mat}], pure = true, tessera.preserves = [{inputs = [0, 1], property = "SPD"}]}
  tessera.define private @lib.inverse(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, {dir = #tessera.dir<out>, type = !mat}], pure = true}
  tessera.define private @lib.inverse_spd(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, {dir = #tessera.dir<out>, type = !mat}], pure = true}

  tessera.optimizations {
    tessera.optimization "if SPD(m), lib.inverse(m) -> lib.inverse_spd(m)"
  }

  // CHECK-LABEL: llvm.func @preserved_through_chain
  // CHECK: tessera.call @lib.inverse_spd
  // CHECK-NOT: tessera.guard
  llvm.func @preserved_through_chain(%n: i64, %s: f64) -> !mat {
    %a = tessera.call @lib.mass(%n) : (i64) -> !mat
    %b = tessera.call @lib.mass(%n) : (i64) -> !mat
    %c = tessera.call @lib.scale(%a, %s) : (!mat, f64) -> !mat
    %d = tessera.call @lib.add(%c, %b) : (!mat, !mat) -> !mat
    %r = tessera.call @lib.inverse(%d) : (!mat) -> !mat
    llvm.return %r : !mat
  }

  // One term of the sum is not known to be SPD, so neither is the sum.
  // CHECK-LABEL: llvm.func @one_input_unknown
  // CHECK: tessera.call @lib.inverse(
  // CHECK-NOT: tessera.guard
  // CHECK-NOT: tessera.call @lib.inverse_spd
  llvm.func @one_input_unknown(%n: i64) -> !mat {
    %a = tessera.call @lib.mass(%n) : (i64) -> !mat
    %k = tessera.call @lib.stiffness(%n) : (i64) -> !mat
    %d = tessera.call @lib.add(%a, %k) : (!mat, !mat) -> !mat
    // expected-remark @below {{optimization rule 'if SPD(m), lib.inverse(m) -> lib.inverse_spd(m)' was not applied here: could not show SPD(m)}}
    %r = tessera.call @lib.inverse(%d) : (!mat) -> !mat
    llvm.return %r : !mat
  }
}

// -----

// A function that writes a matrix and also returns a scalar -- an element
// matrix and its volume, say. The call returns the written matrix first and
// the function's own return value after it, so each guarantee has to land on
// the result it names, not on whatever sits at that position. A property is
// not only for matrices: `positive` here is declared of the scalar.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.assemble(i64, !llvm.ptr {tessera.property = ["SPD"]}) -> (f64 {tessera.property = ["positive"]}) attributes {argModes = [unit, {dir = #tessera.dir<out>, type = !mat}], pure = true}
  tessera.define private @lib.inverse(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, {dir = #tessera.dir<out>, type = !mat}], pure = true}
  tessera.define private @lib.inverse_spd(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, {dir = #tessera.dir<out>, type = !mat}], pure = true}
  tessera.define private @lib.sqrt(f64) -> f64 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.sqrt_positive(f64) -> f64 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.trace(!llvm.ptr) -> f64 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}

  tessera.optimizations {
    tessera.optimization "if SPD(m), lib.inverse(m) -> lib.inverse_spd(m)"
    tessera.optimization "if positive(x), lib.sqrt(x) -> lib.sqrt_positive(x)"
  }

  // CHECK-LABEL: llvm.func @written_argument_and_return
  // CHECK: %[[A:.*]]:2 = tessera.call @lib.assemble
  // CHECK-NEXT: tessera.call @lib.inverse_spd(%[[A]]#0)
  // CHECK-NEXT: tessera.call @lib.sqrt_positive(%[[A]]#1)
  // CHECK-NOT: tessera.guard
  llvm.func @written_argument_and_return(%n: i64) -> f64 {
    %a:2 = tessera.call @lib.assemble(%n) : (i64) -> (!mat, f64)
    %inv = tessera.call @lib.inverse(%a#0) : (!mat) -> !mat
    %root = tessera.call @lib.sqrt(%a#1) : (f64) -> f64
    %t = tessera.call @lib.trace(%inv) : (!mat) -> f64
    %r = llvm.fadd %t, %root : f64
    llvm.return %r : f64
  }
}

// -----

// A declared property implies others: SPD is symmetric, so a rule on
// `symmetric` needs no check for a matrix guaranteed SPD. Without the
// guarantee, `positive_definite` -- which has a runtime check -- still gets a
// guard, where `SPD` would not apply at all.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.mass(!llvm.ptr {llvm.sret = !mat, tessera.property = ["SPD"]}, i64) attributes {argModes = [unit], pure = true}
  tessera.define private @lib.stiffness(!llvm.ptr {llvm.sret = !mat}, i64) attributes {argModes = [unit], pure = true}
  tessera.define private @lib.eig(!llvm.ptr) -> f64 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}
  tessera.define private @lib.eig_sym(!llvm.ptr) -> f64 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}
  tessera.define private @lib.solve(!llvm.ptr) -> f64 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}
  tessera.define private @lib.solve_cg(!llvm.ptr) -> f64 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}

  tessera.optimizations {
    tessera.optimization "if symmetric(m), lib.eig(m) -> lib.eig_sym(m)"
    tessera.optimization "if positive_definite(m), lib.solve(m) -> lib.solve_cg(m)"
  }

  // CHECK-LABEL: llvm.func @implied
  // CHECK: tessera.call @lib.eig_sym
  // CHECK: tessera.call @lib.solve_cg
  // CHECK-NOT: tessera.guard
  llvm.func @implied(%n: i64) -> f64 {
    %m = tessera.call @lib.mass(%n) : (i64) -> !mat
    %e = tessera.call @lib.eig(%m) : (!mat) -> f64
    %s = tessera.call @lib.solve(%m) : (!mat) -> f64
    %r = llvm.fadd %e, %s : f64
    llvm.return %r : f64
  }

  // CHECK-LABEL: llvm.func @runtime_fallback
  // CHECK: tessera.guard "positive_definite(m)"
  llvm.func @runtime_fallback(%n: i64) -> f64 {
    %m = tessera.call @lib.stiffness(%n) : (i64) -> !mat
    %s = tessera.call @lib.solve(%m) : (!mat) -> f64
    llvm.return %s : f64
  }
}

// -----

// Only what the IR does not settle is left for the guard to test.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.mass(!llvm.ptr {llvm.sret = !mat, tessera.property = ["SPD"]}, i64) attributes {argModes = [unit], pure = true}
  tessera.define private @lib.solve(!llvm.ptr, i64) -> f64 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, unit], pure = true}
  tessera.define private @lib.solve_blocked(!llvm.ptr, i64) -> f64 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, unit], pure = true}

  tessera.optimizations {
    tessera.optimization "if SPD(m) && n > 64, lib.solve(m, n) -> lib.solve_blocked(m, n)"
  }

  // CHECK-LABEL: llvm.func @residual
  // CHECK: tessera.guard "n > 64"
  llvm.func @residual(%k: i64, %n: i64) -> f64 {
    %m = tessera.call @lib.mass(%k) : (i64) -> !mat
    %s = tessera.call @lib.solve(%m, %n) : (!mat, i64) -> f64
    llvm.return %s : f64
  }
}

// -----

// A library handle -- a PETSc Mat -- is an opaque pointer to an object whose
// entries are out of reach, so the tessera ops take it as it is and the
// property rides on the handle. PETSc hands a new Mat back through a `Mat *`
// out-argument, which lifts to a result holding the handle.
module {
  tessera.define private @petsc.assemble_mass(i64, !llvm.ptr {tessera.property = ["SPD"]}) -> i32 attributes {argModes = [unit, {dir = #tessera.dir<out>, type = !llvm.ptr}], pure = false}
  tessera.define private @petsc.solve_gmres(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit, unit], pure = false}
  tessera.define private @petsc.solve_cg(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "if SPD(A), petsc.solve_gmres(A, b, x) -> petsc.solve_cg(A, b, x)"
  }

  // CHECK-LABEL: llvm.func @petsc_handle
  // CHECK: %[[A:.*]]:2 = tessera.call @petsc.assemble_mass
  // CHECK-NEXT: tessera.call @petsc.solve_cg(%[[A]]#0, %arg1, %arg2)
  // CHECK-NOT: tessera.guard
  llvm.func @petsc_handle(%n: i64, %b: !llvm.ptr, %x: !llvm.ptr) -> i32 {
    %A:2 = tessera.call @petsc.assemble_mass(%n) : (i64) -> (!llvm.ptr, i32)
    %e = tessera.call @petsc.solve_gmres(%A#0, %b, %x) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e : i32
  }
}

// -----

// A function need not be a tessera op to declare what it returns.
module {
  llvm.func @element_volume(i64) -> (f64 {tessera.property = ["positive"]})
  tessera.define private @lib.sqrt(f64) -> f64 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.sqrt_positive(f64) -> f64 attributes {argModes = [unit], pure = true}

  tessera.optimizations {
    tessera.optimization "if positive(x), lib.sqrt(x) -> lib.sqrt_positive(x)"
  }

  // CHECK-LABEL: llvm.func @plain_call
  // CHECK: tessera.call @lib.sqrt_positive
  llvm.func @plain_call(%n: i64) -> f64 {
    %v = llvm.call @element_volume(%n) : (i64) -> f64
    %r = tessera.call @lib.sqrt(%v) : (f64) -> f64
    llvm.return %r : f64
  }
}

// -----

// A function can assume something of a parameter: every caller promises the
// matrix M refers to is SPD. Inside, the matrix loaded from M is SPD.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.inverse(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, {dir = #tessera.dir<out>, type = !mat}], pure = true}
  tessera.define private @lib.inverse_spd(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, {dir = #tessera.dir<out>, type = !mat}], pure = true}

  tessera.optimizations {
    tessera.optimization "if SPD(m), lib.inverse(m) -> lib.inverse_spd(m)"
  }

  // CHECK-LABEL: llvm.func @assumed_parameter
  // CHECK: tessera.call @lib.inverse_spd
  llvm.func @assumed_parameter(%M: !llvm.ptr {tessera.property = ["SPD"]}) -> !mat {
    %m = llvm.load %M : !llvm.ptr -> !mat
    %r = tessera.call @lib.inverse(%m) : (!mat) -> !mat
    llvm.return %r : !mat
  }

  // If the function could have changed the matrix first -- here by writing
  // through M -- the assumption says nothing about what is loaded after.
  // CHECK-LABEL: llvm.func @written_before_load
  // CHECK: tessera.call @lib.inverse(
  llvm.func @written_before_load(%M: !llvm.ptr {tessera.property = ["SPD"]}, %other: !mat) -> !mat {
    llvm.store %other, %M : !mat, !llvm.ptr
    %m = llvm.load %M : !llvm.ptr -> !mat
    // expected-remark @below {{was not applied here: could not show SPD(m)}}
    %r = tessera.call @lib.inverse(%m) : (!mat) -> !mat
    llvm.return %r : !mat
  }
}

// -----

// Not being able to show a property is not the same as knowing it is absent,
// so a negated property that cannot be shown does not apply either.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.stiffness(!llvm.ptr {llvm.sret = !mat}, i64) attributes {argModes = [unit], pure = true}
  tessera.define private @lib.solve(!llvm.ptr) -> f64 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}
  tessera.define private @lib.solve_general(!llvm.ptr) -> f64 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}

  tessera.optimizations {
    tessera.optimization "if !SPD(m), lib.solve(m) -> lib.solve_general(m)"
  }

  // CHECK-LABEL: llvm.func @negated
  // CHECK: tessera.call @lib.solve(
  // CHECK-NOT: tessera.guard
  llvm.func @negated(%n: i64) -> f64 {
    %m = tessera.call @lib.stiffness(%n) : (i64) -> !mat
    // expected-remark @below {{was not applied here: could not show SPD(m)}}
    %s = tessera.call @lib.solve(%m) : (!mat) -> f64
    llvm.return %s : f64
  }
}
