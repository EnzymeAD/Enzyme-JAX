// RUN: enzymexlamlir-opt %s --convert-llvm-to-cf | FileCheck %s
// RUN: enzymexlamlir-opt %s --canonicalize-scf-for | FileCheck %s
// RUN: enzymexlamlir-opt %s --llvm-to-tessera | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --simplify-affine-exprs | FileCheck %s
// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

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

// Two cleanup landing pads that differ only in the view they store through.
// Merging them makes the invokes pass the views as unwind-destination
// operands, which convert-polygeist-to-llvm cannot lower.

llvm.func @throws()
llvm.func @__gxx_personality_v0(...) -> i32
llvm.func @twin_cleanups(%c: i1, %a: !llvm.ptr, %b: !llvm.ptr) attributes {personality = @__gxx_personality_v0} {
  %zero = arith.constant 0 : i8
  %ma = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xi8>
  %mb = "enzymexla.pointer2memref"(%b) : (!llvm.ptr) -> memref<?xi8>
  llvm.cond_br %c, ^bb1, ^bb2
^bb1:
  llvm.invoke @throws() to ^ret unwind ^lp1 : () -> ()
^lp1:
  %l1 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  affine.store %zero, %ma[0] : memref<?xi8>
  llvm.br ^ret
^bb2:
  llvm.invoke @throws() to ^ret unwind ^lp2 : () -> ()
^lp2:
  %l2 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  affine.store %zero, %mb[0] : memref<?xi8>
  llvm.br ^ret
^ret:
  llvm.return
}

// CHECK-LABEL: llvm.func @twin_tails
// CHECK: ^bb1:
// CHECK: llvm.call @use
// CHECK: ^bb2:
// CHECK: llvm.call @use
// CHECK: ^bb3:

// CHECK-LABEL: llvm.func @twin_cleanups(%arg0: i1, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {personality = @__gxx_personality_v0} {
// CHECK-NEXT:    %c0_i8 = arith.constant 0 : i8
// CHECK-NEXT:    %0 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xi8>
// CHECK-NEXT:    %1 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?xi8>
// CHECK-NEXT:    {{(llvm|cf)}}.cond_br %arg0, ^bb1, ^bb3
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    llvm.invoke @throws() to ^bb5 unwind ^bb2 : () -> ()
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    %2 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
// CHECK-NEXT:    affine.store %c0_i8, %0[0] : memref<?xi8>
// CHECK-NEXT:    {{(llvm|cf)}}.br ^bb5
// CHECK-NEXT:  ^bb3:  // pred: ^bb0
// CHECK-NEXT:    llvm.invoke @throws() to ^bb5 unwind ^bb4 : () -> ()
// CHECK-NEXT:  ^bb4:  // pred: ^bb3
// CHECK-NEXT:    %3 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
// CHECK-NEXT:    affine.store %c0_i8, %1[0] : memref<?xi8>
// CHECK-NEXT:    {{(llvm|cf)}}.br ^bb5
// CHECK-NEXT:  ^bb5:  // 4 preds: ^bb1, ^bb2, ^bb3, ^bb4
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
