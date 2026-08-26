// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" | FileCheck %s

module {
  func.func @if_pure_symbol(%ptr: !llvm.ptr, %n: i32, %c: i1, %c2: i1) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %m = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xi32>
    %outer = scf.if %c2 -> (i32) {
      %if = scf.if %c -> (i32) {
        scf.yield %c0_i32 : i32
      } else {
        %sq = arith.muli %n, %n : i32
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

// Companion to affinecfg_if_side_effect_symbol.mlir: with a read-only body the
// conditional itself is hoistable, so fix() takes its usual path and clones the
// whole scf.if to the top level of the affine scope rather than materializing a
// select.  Either way the load is raised.

// CHECK:         #map = affine_map<()[s0, s1] -> (s0 - s1)>
// CHECK-LABEL:   func.func @if_pure_symbol(
// CHECK-SAME:                              %[[PTR:.+]]: !llvm.ptr, %[[N:.+]]: i32, %[[C:.+]]: i1, %[[C2:.+]]: i1) -> i32 {
// CHECK:           %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK:           %[[NI:.+]] = arith.index_cast %[[N]] : i32 to index
// CHECK:           %[[IF:.+]] = scf.if %[[C]] -> (i32) {
// CHECK:             scf.yield %[[C0_I32]] : i32
// CHECK:           } else {
// CHECK:             %[[SQ:.+]] = arith.muli %[[N]], %[[N]] : i32
// CHECK:             scf.yield %[[SQ]] : i32
// CHECK:           }
// CHECK:           %[[IFI:.+]] = arith.index_cast %[[IF]] : i32 to index
// CHECK:           %[[MEM:.+]] = "enzymexla.pointer2memref"(%[[PTR]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK:           %[[OUTER:.+]] = scf.if %[[C2]] -> (i32) {
// CHECK:             %[[APP:.+]] = affine.apply #map(){{\[}}%[[NI]], %[[IFI]]]
// CHECK:             %[[LOAD:.+]] = memref.load %[[MEM]]{{\[}}%[[APP]]] : memref<?xi32>
// CHECK:             scf.yield %[[LOAD]] : i32
// CHECK:           } else {
// CHECK:             scf.yield %[[C0_I32]] : i32
// CHECK:           }
// CHECK:           return %[[OUTER]] : i32
// CHECK:         }
