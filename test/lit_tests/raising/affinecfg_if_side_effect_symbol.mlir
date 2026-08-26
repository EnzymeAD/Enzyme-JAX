// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" | FileCheck %s

module {
  func.func @if_side_effect_symbol(%ptr: !llvm.ptr, %n: i32, %c: i1, %c2: i1) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %m = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xi32>
    %outer = scf.if %c2 -> (i32) {
      %if = scf.if %c -> (i32) {
        scf.yield %c0_i32 : i32
      } else {
        %sq = arith.muli %n, %n : i32
        %si = arith.index_cast %sq : i32 to index
        memref.store %sq, %m[%si] : memref<?xi32>
        scf.yield %sq : i32
      }
      %ni = arith.index_cast %n : i32 to index
      %ifi = arith.index_cast %if : i32 to index
      %idx = arith.subi %ni, %ifi : index
      %l = memref.load %m[%idx] : memref<?xi32>
      scf.yield %l : i32
    } else {
      scf.yield %c0_i32 : i32
    }
    return %outer : i32
  }
}

// The inner scf.if yields valid symbols, so its result is one -- but the else
// region stores to memory, so AffineApplyNormalizer::fix cannot hoist the
// conditional itself out of the control flow guarding it.  It rewrites the
// value instead, as select(%c, 0, %n * %n), and the stores stay put.  Cloning
// the conditional wholesale was the only strategy fix() had, so it returned
// null and the caller's llvm_unreachable -- UB under NDEBUG -- fell through
// into renumberOneSymbol with a null Value and segfaulted.

// CHECK:         #map = affine_map<()[s0, s1] -> (s0 - s1)>
// CHECK-LABEL:   func.func @if_side_effect_symbol(
// CHECK-SAME:                                     %[[PTR:.+]]: !llvm.ptr, %[[N:.+]]: i32, %[[C:.+]]: i1, %[[C2:.+]]: i1) -> i32 {
// CHECK:           %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK:           %[[NI:.+]] = arith.index_cast %[[N]] : i32 to index
// CHECK:           %[[SQ:.+]] = arith.muli %[[N]], %[[N]] : i32
// CHECK:           %[[SEL:.+]] = arith.select %[[C]], %[[C0_I32]], %[[SQ]] : i32
// CHECK:           %[[SELI:.+]] = arith.index_cast %[[SEL]] : i32 to index
// CHECK:           %[[MEM:.+]] = "enzymexla.pointer2memref"(%[[PTR]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK:           %[[OUTER:.+]] = scf.if %[[C2]] -> (i32) {
// CHECK:             %{{.+}} = scf.if %[[C]] -> (i32) {
// CHECK:               scf.yield %[[C0_I32]] : i32
// CHECK:             } else {
// CHECK:               %[[SI:.+]] = arith.index_cast %[[SQ]] : i32 to index
// CHECK:               memref.store %[[SQ]], %[[MEM]]{{\[}}%[[SI]]] : memref<?xi32>
// CHECK:               scf.yield %[[SQ]] : i32
// CHECK:             }
// CHECK:             %[[APP:.+]] = affine.apply #map(){{\[}}%[[NI]], %[[SELI]]]
// CHECK:             %[[LOAD:.+]] = memref.load %[[MEM]]{{\[}}%[[APP]]] : memref<?xi32>
// CHECK:             scf.yield %[[LOAD]] : i32
// CHECK:           } else {
// CHECK:             scf.yield %[[C0_I32]] : i32
// CHECK:           }
// CHECK:           return %[[OUTER]] : i32
// CHECK:         }
