// RUN: enzymexlamlir-opt --libdevice-funcs-raise --split-input-file %s | FileCheck %s

// An active return is differentiated from a seed, and enzyme.autodiff takes one
// operand for each. __enzyme_autodiff does not name them -- the C interface
// seeds every active return with one, which is what makes its result the
// gradient rather than a directional derivative -- so the raising has to say
// the one itself.

module {
  llvm.mlir.global external constant @enzyme_const() : i8

  llvm.func @sq(%x: f64) -> f64 {
    %r = arith.mulf %x, %x : f64
    llvm.return %r : f64
  }

  llvm.func @__enzyme_autodiff(...) -> f64

  llvm.func @dsq(%x: f64) -> f64 {
    %f = llvm.mlir.addressof @sq : !llvm.ptr
    %r = llvm.call @__enzyme_autodiff(%f, %x) vararg(!llvm.func<f64 (...)>) : (!llvm.ptr, f64) -> f64
    llvm.return %r : f64
  }
}

// CHECK-LABEL: llvm.func @dsq
// CHECK:         %[[one:.+]] = arith.constant 1.000000e+00 : f64
// CHECK:         enzyme.autodiff @sq(%arg0, %[[one]])
// CHECK-SAME:      activity = [#enzyme<activity enzyme_active>]
// CHECK-SAME:      ret_activity = [#enzyme<activity enzyme_active>]

// -----

// Forward mode was not raised at all. MFEM writes its calls in the interleaved
// form: one sticky activity marker, then every primal, then enzyme_interleave,
// then every shadow in the same order, then whole-call markers.

module {
  llvm.mlir.global external constant @enzyme_dup() : i8
  llvm.mlir.global external constant @enzyme_interleave() : i8
  llvm.mlir.global external constant @enzyme_runtime_activity() : i8

  llvm.func @kern(%a: !llvm.ptr, %b: !llvm.ptr) {
    llvm.return
  }

  llvm.func @__enzyme_fwddiff(...)

  llvm.func @dkern(%a: !llvm.ptr, %da: !llvm.ptr, %b: !llvm.ptr, %db: !llvm.ptr) {
    %f = llvm.mlir.addressof @kern : !llvm.ptr
    %dupaddr = llvm.mlir.addressof @enzyme_dup : !llvm.ptr
    %dup = llvm.load %dupaddr : !llvm.ptr -> i8
    %ilvaddr = llvm.mlir.addressof @enzyme_interleave : !llvm.ptr
    %ilv = llvm.load %ilvaddr : !llvm.ptr -> i8
    %rtaaddr = llvm.mlir.addressof @enzyme_runtime_activity : !llvm.ptr
    %rta = llvm.load %rtaaddr : !llvm.ptr -> i8
    llvm.call @__enzyme_fwddiff(%f, %dup, %a, %b, %ilv, %da, %db, %rta) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i8, !llvm.ptr, !llvm.ptr, i8, !llvm.ptr, !llvm.ptr, i8) -> ()
    llvm.return
  }
}

// Each primal is paired with the shadow that sat at the same place after the
// separator, and the sticky marker covers both.

// CHECK-LABEL: llvm.func @dkern
// CHECK:         enzyme.fwddiff @kern(%arg0, %arg1, %arg2, %arg3)
// CHECK-SAME:      activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>]
// CHECK-SAME:      ret_activity = []
