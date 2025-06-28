// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(llvm.func(canonicalize-loops))" %s | FileCheck %s

module {
  llvm.func local_unnamed_addr @f(%arg0: i32, %arg1: !llvm.ptr) {
    %cst = arith.constant 0.000000e+00 : f64
    %2 = arith.index_castui %arg0 : i32 to index
    scf.index_switch %2 
    case 0 {
      llvm.store %cst, %arg1 {alignment = 8 : i64} : f64, !llvm.ptr
      scf.yield
    }
    default {
    }
    llvm.return
  }
}

// CHECK:  llvm.func local_unnamed_addr @f(%arg0: i32, %arg1: !llvm.ptr) {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.index_castui %arg0 : i32 to index
// CHECK-NEXT:    %1 = arith.cmpi eq, %0, %c0 : index
// CHECK-NEXT:    scf.if %1 {
// CHECK-NEXT:      llvm.store %cst, %arg1 {alignment = 8 : i64} : f64, !llvm.ptr
// CHECK-NEXT:    }
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
