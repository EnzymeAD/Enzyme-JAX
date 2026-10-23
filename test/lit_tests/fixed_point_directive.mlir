// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-llvm-to-cf,enzyme-lift-cf-to-scf,canonicalize,apply-fixed-point-directives)" | FileCheck %s

// A kernel loop that marks itself with `__enzyme_set_fixed_point(reduction,
// max_iters, control)` (LLVM Enzyme's marker, which Reactant's
// `fixed_point_implicit` emits): the loop is marked with those settings
// whatever it looks like, a negative setting selects the default, and the
// call and its declaration go away. A loop without the call is left alone.
llvm.func @__enzyme_set_fixed_point(f64, i64, !llvm.ptr)

llvm.func @marked(%x0: f64, %a: f64, %tol: f64) -> f64 {
  %two = llvm.mlir.constant(2.0 : f64) : f64
  %red = llvm.mlir.constant(0.25 : f64) : f64
  %max = llvm.mlir.constant(30 : i64) : i64
  %null = llvm.mlir.zero : !llvm.ptr
  llvm.br ^bb1(%x0 : f64)
^bb1(%x: f64):
  llvm.call @__enzyme_set_fixed_point(%red, %max, %null) : (f64, i64, !llvm.ptr) -> ()
  %x2 = llvm.fmul %x, %x : f64
  %r = llvm.fsub %x2, %a : f64
  %d = llvm.fmul %two, %x : f64
  %q = llvm.fdiv %r, %d : f64
  %xn = llvm.fsub %x, %q : f64
  %aq = llvm.intr.fabs(%q) : (f64) -> f64
  %go = llvm.fcmp "ogt" %aq, %tol : f64
  llvm.cond_br %go, ^bb1(%xn : f64), ^bb2(%xn : f64)
^bb2(%res: f64):
  llvm.return %res : f64
}

llvm.func @defaults(%x0: f64, %a: f64, %tol: f64) -> f64 {
  %two = llvm.mlir.constant(2.0 : f64) : f64
  %red = llvm.mlir.constant(-1.0 : f64) : f64
  %max = llvm.mlir.constant(-1 : i64) : i64
  %null = llvm.mlir.zero : !llvm.ptr
  llvm.br ^bb1(%x0 : f64)
^bb1(%x: f64):
  llvm.call @__enzyme_set_fixed_point(%red, %max, %null) : (f64, i64, !llvm.ptr) -> ()
  %x2 = llvm.fmul %x, %x : f64
  %r = llvm.fsub %x2, %a : f64
  %d = llvm.fmul %two, %x : f64
  %q = llvm.fdiv %r, %d : f64
  %xn = llvm.fsub %x, %q : f64
  %aq = llvm.intr.fabs(%q) : (f64) -> f64
  %go = llvm.fcmp "ogt" %aq, %tol : f64
  llvm.cond_br %go, ^bb1(%xn : f64), ^bb2(%xn : f64)
^bb2(%res: f64):
  llvm.return %res : f64
}

llvm.func @unmarked(%x0: f64, %a: f64, %tol: f64) -> f64 {
  %two = llvm.mlir.constant(2.0 : f64) : f64
  llvm.br ^bb1(%x0 : f64)
^bb1(%x: f64):
  %x2 = llvm.fmul %x, %x : f64
  %r = llvm.fsub %x2, %a : f64
  %d = llvm.fmul %two, %x : f64
  %q = llvm.fdiv %r, %d : f64
  %xn = llvm.fsub %x, %q : f64
  %aq = llvm.intr.fabs(%q) : (f64) -> f64
  %go = llvm.fcmp "ogt" %aq, %tol : f64
  llvm.cond_br %go, ^bb1(%xn : f64), ^bb2(%xn : f64)
^bb2(%res: f64):
  llvm.return %res : f64
}

// CHECK-NOT:   __enzyme_set_fixed_point
// CHECK-LABEL: llvm.func @marked(
// CHECK:         scf.while
// CHECK:         } attributes {enzyme.fixed_point, enzyme.fixed_point_max_iters = 30 : i64, enzyme.fixed_point_reduction = 2.500000e-01 : f64}
// CHECK-LABEL: llvm.func @defaults(
// CHECK:         scf.while
// CHECK:         } attributes {enzyme.fixed_point}
// CHECK-LABEL: llvm.func @unmarked(
// CHECK:         scf.while
// CHECK-NOT:     enzyme.fixed_point
