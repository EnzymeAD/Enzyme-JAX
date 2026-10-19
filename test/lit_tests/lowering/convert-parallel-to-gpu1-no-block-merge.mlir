// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// Aggressive region simplification (the greedy driver's default) merges
// identical blocks by adding their differing values as block arguments and
// appending them to every predecessor's successor operands. One of these
// predecessors is an llvm.invoke, whose successor operands must be LLVM
// types: merging the two store blocks would hand it an index, which the op
// cannot carry. The pass's drivers keep block merging off, so both blocks
// survive and the invoke keeps zero successor operands.

module {
  llvm.func @g()
  llvm.func @pers(i32) -> i32
  llvm.func @f(%p: !llvm.ptr, %v: f64, %c: i1) attributes {personality = @pers} {
    %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    llvm.cond_br %c, ^inv, ^direct
  ^inv:
    llvm.invoke @g() to ^n unwind ^u : () -> ()
  ^n:
    memref.store %v, %m[%c1] : memref<?xf64>
    llvm.br ^end
  ^direct:
    llvm.br ^n2
  ^n2:
    memref.store %v, %m[%c2] : memref<?xf64>
    llvm.br ^end
  ^end:
    llvm.return
  ^u:
    %lp = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @f
// CHECK: llvm.invoke @g() to ^[[N:.+]] unwind
// CHECK: ^[[N]]:
// CHECK-NEXT: memref.store
// CHECK: memref.store
