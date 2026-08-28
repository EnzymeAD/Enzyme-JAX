// RUN: enzymexlamlir-opt %s --convert-llvm-to-cf | FileCheck %s
// RUN: enzymexlamlir-opt %s --canonicalize-scf-for | FileCheck %s
// RUN: enzymexlamlir-opt %s --llvm-to-tessera | FileCheck %s

// Cold error tails that are structurally identical modulo constants must not
// be merged: LLVM cannot split a shared tail apart again, and its machine
// passes hoist the merged tail's setup into the hot path, taxing every call.

llvm.func @use(i64)
llvm.func @twin_tails(%c: i1) {
  llvm.cond_br %c, ^bb1, ^bb2
^bb1:
  %a = llvm.mlir.constant(27 : i64) : i64
  llvm.call @use(%a) : (i64) -> ()
  llvm.br ^bb3
^bb2:
  %b = llvm.mlir.constant(20 : i64) : i64
  llvm.call @use(%b) : (i64) -> ()
  llvm.br ^bb3
^bb3:
  llvm.return
}

// CHECK-LABEL: llvm.func @twin_tails
// CHECK: ^bb1:
// CHECK-NEXT: llvm.call @use
// CHECK: ^bb2:
// CHECK-NEXT: llvm.call @use
// CHECK: ^bb3:
