// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// `-1 - %n` is a valid affine symbol wherever it is defined (a pure function
// of symbols), so the normalizer used to leave it inside the scf.if; the
// select, no symbol with its i1 condition, was then cloned next to it, inside
// the conditional, where it is no symbol either. Both hoist above the
// conditional now.

llvm.func @use(!llvm.ptr)

llvm.func @f(%arg0: !llvm.ptr, %n: i64, %c: i1) {
  %c-1 = arith.constant -1 : index
  %5 = arith.index_cast %n : i64 to index
  scf.if %c {
    %32 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %34 = arith.subi %c-1, %5 : index
    %35 = arith.select %c, %34, %5 : index
    %40 = memref.load %32[%35] : memref<?x!llvm.ptr>
    llvm.call @use(%40) : (!llvm.ptr) -> ()
  }
  llvm.return
}

// CHECK:    llvm.func @use(!llvm.ptr)
// CHECK-NEXT:  llvm.func @f(%arg0: !llvm.ptr, %arg1: i64, %arg2: i1) {
// CHECK-NEXT:    %c-1 = arith.constant -1 : index
// CHECK-NEXT:    %0 = arith.index_cast %arg1 : i64 to index
// CHECK-NEXT:    %1 = arith.subi %c-1, %0 : index
// CHECK-NEXT:    %2 = arith.select %arg2, %1, %0 : index
// CHECK-NEXT:    scf.if %arg2 {
// CHECK-NEXT:      %3 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?x!llvm.ptr>
// CHECK-NEXT:      %4 = affine.load %3[symbol(%2)] : memref<?x!llvm.ptr>
// CHECK-NEXT:      llvm.call @use(%4) : (!llvm.ptr) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
