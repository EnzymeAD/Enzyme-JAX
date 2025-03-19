// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

module {
  llvm.func @f() {
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst_22 = arith.constant 1.000000e-04 : f64
    %c5_i64 = arith.constant 5 : i64
    scf.for %arg16 = %c0_i64 to %c5_i64 step %c1_i64  : i64 {
      llvm.intr.experimental.noalias.scope.decl <id = distinct[4]<>, domain = <id = distinct[5]<>, description = "julia_iterate_interface_fluxes_168833">>
      scf.yield
    }
    llvm.return
  }
}

// CHECK:  llvm.func @f() {
// CHECK-NEXT:    affine.for %arg0 = 0 to 5 {
// CHECK-NEXT:      llvm.intr.experimental.noalias.scope.decl #alias_scope
// CHECK-NEXT:    }
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }