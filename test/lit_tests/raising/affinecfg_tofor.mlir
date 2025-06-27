// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

module {
  llvm.func @for(%arg1 : i32) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg7 = %c0_i32 to %arg1 step %c1_i32  : i32 {
        %139 = arith.extsi %arg7 : i32 to i64
        llvm.call @_ZNSt6vectorIfSaIfEEixEm(%139) {no_unwind} : (i64 {llvm.noundef}) -> ()
    }
    llvm.return
  }
  llvm.func @_ZNSt6vectorIfSaIfEEixEm(%arg1: i64 {llvm.noundef}) {
    llvm.return
  }
}

// CHECK:  llvm.func @for(%arg0: i32) {
// CHECK-NEXT:    %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:    affine.for %arg1 = 0 to %0 {
// CHECK-NEXT:      %1 = arith.index_cast %arg1 : index to i32
// CHECK-NEXT:      %2 = arith.extsi %1 : i32 to i64
// CHECK-NEXT:      llvm.call @_ZNSt6vectorIfSaIfEEixEm(%2) {no_unwind} : (i64 {llvm.noundef}) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
