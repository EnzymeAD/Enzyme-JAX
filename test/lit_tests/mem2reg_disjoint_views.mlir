// RUN: enzymexlamlir-opt %s -polygeist-mem2reg -split-input-file | FileCheck %s

// A store through a view that counts in other units than the load says
// nothing about a field it does not reach: the two are related by where they
// land and how far they reach, not by the units either counted its way in.

func.func @disjoint_byte_store(%b: i8) -> i32 {
  %c1 = arith.constant 1 : i32
  %v = arith.constant 42 : i32
  %a = llvm.alloca %c1 x !llvm.struct<"S", (i32, i32, i32, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %mi = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xi32>
  %mb = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xi8>
  affine.store %v, %mi[1] : memref<?xi32>
  affine.store %b, %mb[12] : memref<?xi8>
  %r = affine.load %mi[1] : memref<?xi32>
  return %r : i32
}

// CHECK-LABEL: func.func @disjoint_byte_store(
// CHECK-NOT: affine.load
// CHECK: return %c42_i32 : i32

// -----

// A byte written inside the field the load reads is a different matter: the
// load cannot be answered from the store before it.

func.func @overlapping_byte_store(%b: i8) -> i32 {
  %c1 = arith.constant 1 : i32
  %v = arith.constant 42 : i32
  %a = llvm.alloca %c1 x !llvm.struct<"S", (i32, i32, i32, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %mi = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xi32>
  %mb = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xi8>
  affine.store %v, %mi[1] : memref<?xi32>
  affine.store %b, %mb[5] : memref<?xi8>
  %r = affine.load %mi[1] : memref<?xi32>
  return %r : i32
}

// CHECK-LABEL: func.func @overlapping_byte_store(
// CHECK: affine.load

// -----

// A load through another view reads and so clobbers nothing, wherever it
// lands.

func.func @disjoint_byte_load() -> (i32, i8) {
  %c1 = arith.constant 1 : i32
  %v = arith.constant 42 : i32
  %a = llvm.alloca %c1 x !llvm.struct<"S", (i32, i32, i32, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %mi = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xi32>
  %mb = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xi8>
  affine.store %v, %mi[1] : memref<?xi32>
  %r = affine.load %mi[1] : memref<?xi32>
  %q = affine.load %mb[12] : memref<?xi8>
  return %r, %q : i32, i8
}

// CHECK-LABEL: func.func @disjoint_byte_load(
// CHECK: %[[q:.+]] = affine.load
// CHECK: return %c42_i32, %[[q]] : i32, i8

// -----

// An index that could be anywhere still blocks the slot it reads, and only
// that one: the constant slots around it forward.

func.func @dynamic_index_load(%b: i8, %d: index) -> (i32, i32) {
  %c1 = arith.constant 1 : i32
  %v = arith.constant 42 : i32
  %a = llvm.alloca %c1 x !llvm.struct<"S", (i32, i32, i32, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %mi = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xi32>
  %mb = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xi8>
  affine.store %v, %mi[1] : memref<?xi32>
  affine.store %b, %mb[12] : memref<?xi8>
  %r = affine.load %mi[1] : memref<?xi32>
  %dyn = memref.load %mi[%d] : memref<?xi32>
  return %r, %dyn : i32, i32
}

// CHECK-LABEL: func.func @dynamic_index_load(
// CHECK: %[[dyn:.+]] = memref.load
// CHECK: return %c42_i32, %[[dyn]] : i32, i32
