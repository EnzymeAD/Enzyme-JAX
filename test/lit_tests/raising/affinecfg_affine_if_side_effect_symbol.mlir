// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" | FileCheck %s

#set = affine_set<()[s0] : (s0 - 1 >= 0)>
module {
  func.func @affine_if_side_effect_symbol(%ptr: !llvm.ptr, %n: i32, %c2: i1) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %m = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xi32>
    %ni = arith.index_cast %n : i32 to index
    %outer = scf.if %c2 -> (i32) {
      %if = affine.if #set()[%ni] -> i32 {
        affine.yield %c0_i32 : i32
      } else {
        %sq = arith.muli %n, %n : i32
        %si = arith.index_cast %sq : i32 to index
        memref.store %sq, %m[%si] : memref<?xi32>
        affine.yield %sq : i32
      }
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

// The affine.if counterpart of affinecfg_if_side_effect_symbol.mlir.  Its
// condition is an integer set rather than an i1, so fix() cannot rewrite the
// value as an arith.select; it materializes a bodiless affine.if over the same
// set instead, yielding the two hoisted arms.  isValidDim accepts valid
// symbols, so the legalized set operands stay legal at the new location.  The
// storing affine.if is left where it is.

// CHECK:         #map = affine_map<()[s0, s1] -> (s0 - s1)>
// CHECK:         #set = affine_set<()[s0] : (s0 - 1 >= 0)>
// CHECK-LABEL:   func.func @affine_if_side_effect_symbol(
// CHECK-SAME:                                            %[[PTR:.+]]: !llvm.ptr, %[[N:.+]]: i32, %[[C2:.+]]: i1) -> i32 {
// CHECK:           %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK:           %[[SQ:.+]] = arith.muli %[[N]], %[[N]] : i32
// CHECK:           %[[MEM:.+]] = "enzymexla.pointer2memref"(%[[PTR]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK:           %[[NI:.+]] = arith.index_cast %[[N]] : i32 to index
// CHECK:           %[[HOISTED:.+]] = affine.if #set(){{\[}}%[[NI]]] -> i32 {
// CHECK:             affine.yield %[[C0_I32]] : i32
// CHECK:           } else {
// CHECK:             affine.yield %[[SQ]] : i32
// CHECK:           }
// CHECK:           %[[HOISTEDI:.+]] = arith.index_cast %[[HOISTED]] : i32 to index
// CHECK:           %[[OUTER:.+]] = scf.if %[[C2]] -> (i32) {
// CHECK:             %{{.+}} = affine.if #set(){{\[}}%[[NI]]] -> i32 {
// CHECK:               affine.yield %[[C0_I32]] : i32
// CHECK:             } else {
// CHECK:               %[[SI:.+]] = arith.index_cast %[[SQ]] : i32 to index
// CHECK:               memref.store %[[SQ]], %[[MEM]]{{\[}}%[[SI]]] : memref<?xi32>
// CHECK:               affine.yield %[[SQ]] : i32
// CHECK:             }
// CHECK:             %[[APP:.+]] = affine.apply #map(){{\[}}%[[NI]], %[[HOISTEDI]]]
// CHECK:             %[[LOAD:.+]] = memref.load %[[MEM]]{{\[}}%[[APP]]] : memref<?xi32>
// CHECK:             scf.yield %[[LOAD]] : i32
// CHECK:           } else {
// CHECK:             scf.yield %[[C0_I32]] : i32
// CHECK:           }
// CHECK:           return %[[OUTER]] : i32
// CHECK:         }
